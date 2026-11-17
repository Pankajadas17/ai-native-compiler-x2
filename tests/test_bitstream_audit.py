import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from verification.bitstream_audit import (FabricProfile, audit_against_fasm, audit_structure, decode_set_bits,
                                          load_spec, parse_fasm, profile_from_fabric_csv, required_bits)

FIX = Path(__file__).parent / "fixtures" / "fabulous_demo"
HDR, ROWS, FRAME = 20, 14, 4 + 14 * 4          # sync header bytes, rows per frame, bytes per frame (select word + data)


@pytest.fixture(scope="module")
def data():
    return (FIX / "design.bin").read_bytes()


@pytest.fixture(scope="module")
def prof():
    return profile_from_fabric_csv((FIX / "fabric.csv").read_text(), "fabulous_demo")


@pytest.fixture(scope="module")
def spec():
    return load_spec(str(FIX / "spec_reduced.json"))


@pytest.fixture(scope="module")
def fasm():
    return (FIX / "design.fasm").read_text()


def failing(rep):
    return {c["id"] for c in rep["checks"] if not c["ok"]}


def at(frame):                                   # byte offset of a frame's select word
    return HDR + frame * FRAME


def patch(b, off, new):
    return b[:off] + new + b[off + len(new):]


def word(b, off):
    return struct.unpack(">I", b[off:off + 4])[0]


# ---------------------------------------------------------------- the fixture is a real, intact FABulous output
def test_fixture_manifest_matches_the_files():
    man = json.loads((FIX / "MANIFEST.json").read_text())
    assert man["tools"]["FABulous"] == "2.2.0" and "yosys" in man["tools"] and "nextpnr-generic" in man["tools"]
    for name, meta in man["files"].items():
        raw = (FIX / name).read_bytes()
        assert len(raw) == meta["size"] and hashlib.sha256(raw).hexdigest() == meta["sha256"], name


def test_profile_comes_from_fabric_csv(prof):
    assert (prof.rows_per_frame, prof.columns, prof.frames_per_col, prof.expected_frames) == (14, 10, 20, 200)


# ---------------------------------------------------------------- structural audit
def test_real_bitstream_passes_every_structural_check(data, prof):
    r = audit_structure(data, prof)
    assert r["ok"] and failing(r) == set()
    s = r["stats"]
    assert (s["bytes"], s["frames"], s["set_bits"], s["zero_frames"]) == (12024, 200, 713, 161)
    assert 14 in s["rows_per_frame_that_parse"]          # consistency only: other geometries can parse by coincidence (many zero frames)


MUTATIONS = {
    "missing desync word":       (lambda b: b[:-4],                                         {"V0.3"}),
    "size not a multiple of 4":  (lambda b: b[:-2],                                         {"V0.1a", "V0.3"}),
    "broken sync word":          (lambda b: patch(b, 16, b"\x00\x00\x00\x00"),              {"V0.2", "V0.3"}),
    "broken preamble":           (lambda b: patch(b, 0, b"\xFF"),                           {"V0.2", "V0.3"}),
    "desync flag cleared":       (lambda b: b[:-4] + b"\x00\x00\x00\x00",                    {"V0.3"}),
    "trailing garbage":          (lambda b: b + b"\xDE\xAD\xBE\xEF",                        {"V0.3b"}),
    "two frames swapped":        (lambda b: b[:at(0)] + b[at(1):at(2)] + b[at(0):at(1)] + b[at(2):],            {"V0.5a"}),
    "frame written twice":       (lambda b: patch(b, at(2), b[at(1):at(2)]),                {"V0.5a", "V0.5b", "V0.5c"}),
    "column out of range":       (lambda b: patch(b, at(0), struct.pack(">I", (31 << 27) | 1)),                 {"V0.4c", "V0.5a"}),
    "multi-hot strobe":          (lambda b: patch(b, at(0), struct.pack(">I", 0b11)),       {"V0.4b", "V0.5a"}),
    "reserved bit set":          (lambda b: patch(b, at(3), struct.pack(">I", word(b, at(3)) | (1 << 21))),     {"V0.4a"}),
}


@pytest.mark.parametrize("label", sorted(MUTATIONS))
def test_structural_audit_catches_each_corruption(data, prof, label):
    mutate, expected = MUTATIONS[label]
    bad = mutate(data)
    assert bad != data
    r = audit_structure(bad, prof)
    assert not r["ok"] and expected <= failing(r), (label, failing(r))


def test_size_limit_and_wrong_geometry_are_reported(data, prof):
    assert "V0.1b" in failing(audit_structure(data, replace(prof, max_bytes=12000)))
    r = audit_structure(data, replace(prof, rows_per_frame=13))
    assert not r["ok"] and {"V0.3", "V0.6"} <= failing(r)


def test_structural_audit_is_blind_to_bit_flips_inside_frame_data(data, prof):
    """An honest limit, locked in by a test: the structural level cannot see a flipped configuration bit."""
    zero = next(k for k in range(200) if all(word(data, at(k) + 4 + 4 * r) == 0 for r in range(ROWS)))
    flipped = patch(data, at(zero) + 4, struct.pack(">I", 1))
    assert audit_structure(flipped, prof)["ok"] is True


# ---------------------------------------------------------------- semantic decode against the FASM
def test_real_bitstream_equals_exactly_what_the_fasm_requires(data, prof, spec, fasm):
    d = audit_against_fasm(data, prof, fasm, spec)
    assert d["ok"] and (d["required_bits"], d["set_bits"], d["missing_count"], d["unexplained_count"]) == (713, 713, 0, 0)
    assert d["fasm_features_not_in_spec"] == [] and d["unparsed_fasm_lines"] == []


def test_a_flipped_payload_bit_is_found_and_located(data, prof, spec, fasm):
    zero = next(k for k in range(200) if all(word(data, at(k) + 4 + 4 * r) == 0 for r in range(ROWS)))
    bad = patch(data, at(zero) + 4, struct.pack(">I", 1))
    d = audit_against_fasm(bad, prof, fasm, spec)
    assert not d["ok"] and d["unexplained_count"] == 1 and d["missing_count"] == 0
    assert d["unexplained"][0]["tile"].startswith("X") and d["unexplained"][0]["spec_owners"]


def test_a_cleared_required_bit_names_the_feature_that_needed_it(data, prof, spec, fasm):
    k = next(k for k in range(200) if any(word(data, at(k) + 4 + 4 * r) for r in range(ROWS)))
    r = next(r for r in range(ROWS) if word(data, at(k) + 4 + 4 * r))
    w = word(data, at(k) + 4 + 4 * r); low = w & -w                       # clear the lowest set bit
    bad = patch(data, at(k) + 4 + 4 * r, struct.pack(">I", w & ~low))
    d = audit_against_fasm(bad, prof, fasm, spec)
    assert not d["ok"] and d["missing_count"] == 1 and d["missing"][0]["needed_by"]


def test_swapped_frames_fail_both_levels(data, prof, spec, fasm):
    bad = data[:at(0)] + data[at(1):at(2)] + data[at(0):at(1)] + data[at(2):]
    assert not audit_structure(bad, prof)["ok"]


def test_fasm_that_does_not_match_the_bitstream_is_detected(data, prof, spec, fasm):
    d = audit_against_fasm(data, prof, fasm + "\nX3Y5.NOPE.NOPE\n", spec)               # a feature the spec does not know
    assert not d["ok"] and d["fasm_features_not_in_spec"] == ["X3Y5.NOPE.NOPE"]
    lines = [l for l in fasm.split("\n") if l.strip() and not l.startswith("#")]
    dropped = "\n".join(l for l in fasm.split("\n") if l != lines[0])
    d2 = audit_against_fasm(data, prof, dropped, spec)
    assert not d2["ok"] and d2["unexplained_count"] >= 1


def test_unparsable_fasm_lines_are_reported_not_ignored(data, prof, spec):
    d = audit_against_fasm(data, prof, "X0Y1.W2MID5.A_I\nthis is not fasm\n", spec)
    assert not d["ok"] and d["unparsed_fasm_lines"] == ["this is not fasm"]


def test_alias_for_bit_zero_of_a_vector_resolves(spec):
    req, unknown, bad = required_bits("X1Y12.C.INIT[15:0] = 16'b0000000000000001\n", spec)
    assert unknown == [] and ("X1Y12", 488) in req            # spec spells bit 0 as C.INIT (no index)


def test_fasm_literals_are_parsed_in_binary_hex_and_decimal():
    f, bad = parse_fasm("X1Y1.A.INIT[7:0] = 8'b00001111\nX1Y1.B.INIT[7:0] = 8'h0F\nX1Y1.C.INIT[7:0] = 8'd15\nX1Y1.D.E\n# comment\n")
    assert bad == [] and [x[3] for x in f[:3]] == ["00001111"] * 3 and f[3][2] is None


def test_decode_of_a_non_parsing_stream_is_reported_cleanly(prof, spec, fasm):
    d = audit_against_fasm(b"\x00" * 24, prof, fasm, spec)
    assert not d["ok"] and "does not parse" in d["error"]


def test_layout_assumption_is_recorded_with_its_evidence(prof):
    assert "713/713" in prof.layout["verified_on"] and prof.layout["row_word"] == "rows_per_frame - Y"
