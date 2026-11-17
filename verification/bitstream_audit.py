"""
V0 / S10: bitstream audit for FABulous `.bin` files. Deterministic, read-only, no FABulous installation needed.

Wire format (normative source: fabulous_bit_gen/bit_gen.py, FABulous 2.2.0):
  [20-byte sync header 00AAFF01 00000001 00000000 00000000 FAB0FAB1]
  for column 0..columns-1, for frame 0..frames_per_col-1:
      [frame-select word, 32-bit big-endian]  bits 31:27 column | 26:21 reserved (0) | 20 desync flag | 19:0 one-hot frame strobe
      [rows_per_frame data words]             (interior rows only by default, Y descending)
  [desync word = 1 << 20]

Two levels:
  audit_structure       needs only the .bin + FabricProfile            (does NOT see bit flips inside frame data)
  audit_against_fasm    also needs the FASM and the fabric spec        (decodes the bits and compares them with the FASM)

LAYOUT ASSUMPTION (verified on the FABulous 2.2.0 demo fabric: sequential_16bit_en 713/713 bits, add8_sat_fab 653/653 bits): tile (X, Y) configuration bit `idx` lives in
column X, frame idx // 32, row word (rows_per_frame - Y), bit idx % 32. It is stored in FabricProfile.layout and must be
re-verified for any other fabric before the decode check is trusted.

BM Labs chip (classic_fabric_reram, sky130-reram-fpga): border rows are INCLUDED (spec ArchSpecs IncludeBorderRows),
so the first row word is Y = rows_per_frame - 1 (FabricProfile.top_row_y). Verified 2026-11-12 on 7 designs, every
required bit present and no other bit set; see chip/chip_bitstream.py.
"""
import csv
import hashlib
import io
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

SYNC_HEADER_HEX = "00AAFF01000000010000000000000000FAB0FAB1"
DEFAULT_LAYOUT = {"column": "X", "frame": "idx // 32", "row_word": "rows_per_frame - Y", "bit": "idx % 32",
                  "verified_on": "FABulous 2.2.0 demo fabric: sequential_16bit_en 713/713 and add8_sat_fab 653/653 required bits"}


@dataclass(frozen=True)
class FabricProfile:
    name: str
    rows_per_frame: int
    columns: int
    frames_per_col: int = 20
    frame_bits_per_row: int = 32
    max_bytes: int = 16384                    # FABulous simulation-flow default (MAX_BITBYTES)
    sync_header_hex: str = SYNC_HEADER_HEX
    desync_bit: int = 20
    column_shift: int = 27
    layout: dict = field(default_factory=lambda: dict(DEFAULT_LAYOUT), compare=False)
    # Y of the tile in the FIRST row word of a frame. None = rows_per_frame (FABulous 2.2.0 demo flow:
    # border rows excluded). The BM Labs chip (classic_fabric_reram, border rows INCLUDED) uses rows_per_frame - 1.
    top_row_y: int = None

    @property
    def expected_frames(self):
        return self.columns * self.frames_per_col


class BitstreamFormatError(Exception):
    pass


def profile_from_fabric_csv(text, name="fabric", border_rows=2):
    """Derive geometry from FABulous's fabric.csv. ASSUMPTION: the two border rows are excluded from the bitstream
    (FABulous default; its log says 'Border rows excluded from bitstream (default)')."""
    rows, inside = [], False
    for r in csv.reader(io.StringIO(text)):
        if r and r[0].strip() == "FabricBegin":
            inside = True; continue
        if r and r[0].strip() == "FabricEnd":
            break
        if inside and r and r[0].strip() and not r[0].startswith("#"):
            n = 0
            for cell in r:
                if cell.strip() == "" or cell.strip().startswith("#"):
                    break
                n += 1
            rows.append(n)
    if not rows:
        raise ValueError("no FabricBegin/FabricEnd section found")
    return FabricProfile(name=name, rows_per_frame=len(rows) - border_rows, columns=max(rows))


# ----------------------------------------------------------------------------------------- parsing
def _words(data):
    return struct.unpack(f">{len(data) // 4}I", data)


def parse_frames(data, p, rows_per_frame=None):
    """Return (frames, trailing_bytes). frames = [(column, strobe, rows[list of words])]; raises BitstreamFormatError."""
    rows = p.rows_per_frame if rows_per_frame is None else rows_per_frame
    hdr = bytes.fromhex(p.sync_header_hex)
    if len(data) % 4:
        raise BitstreamFormatError(f"size {len(data)} is not a multiple of 4 bytes")
    if data[:len(hdr)] != hdr:
        raise BitstreamFormatError("sync header mismatch: expected " + p.sync_header_hex + ", found " + data[:len(hdr)].hex().upper())
    w, i, frames = _words(data), len(hdr) // 4, []
    desync_mask = 1 << p.desync_bit
    while True:
        if i >= len(w):
            raise BitstreamFormatError("stream ended without a desync word")
        h = w[i]; i += 1
        if h & desync_mask:
            if h != desync_mask:
                raise BitstreamFormatError(f"desync word has extra bits set: 0x{h:08X}")
            return frames, len(w) - i
        if i + rows > len(w):
            raise BitstreamFormatError(f"truncated frame {len(frames)}: needs {rows} data words, only {len(w) - i} left")
        frames.append(((h >> p.column_shift) & 0x1F, h & (desync_mask - 1), list(w[i:i + rows]), h))
        i += rows


def _chk(i, title, ok, detail="", sev="error"):
    return {"id": i, "title": title, "ok": bool(ok), "severity": sev, "detail": detail}


def audit_structure(data, p):
    checks, stats = [], {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    checks.append(_chk("V0.1a", "size is a whole number of 32-bit words", len(data) % 4 == 0, f"{len(data)} bytes"))
    checks.append(_chk("V0.1b", f"size within the {p.max_bytes}-byte limit", len(data) <= p.max_bytes,
                       f"{len(data)} bytes (limit is the FABulous simulation-flow default)"))
    hdr = bytes.fromhex(p.sync_header_hex)
    checks.append(_chk("V0.2", "20-byte sync header is exact", data[:len(hdr)] == hdr, "found " + data[:len(hdr)].hex().upper()))
    frames, trailing, err = [], 0, None
    try:
        frames, trailing = parse_frames(data, p)
    except BitstreamFormatError as e:
        err = str(e)
    checks.append(_chk("V0.3", f"frames parse with rows_per_frame={p.rows_per_frame} and end in a desync word", err is None, err or f"{len(frames)} frames"))
    if err is None:
        checks.append(_chk("V0.3b", "nothing follows the desync word", trailing == 0, f"{trailing} trailing word(s)"))
        bad_res = [k for k, f in enumerate(frames) if f[3] & 0x07E00000]              # bits 26:21
        checks.append(_chk("V0.4a", "reserved header bits 26:21 are zero", not bad_res, f"frames {bad_res[:8]}" if bad_res else ""))
        bad_hot = [k for k, f in enumerate(frames) if bin(f[1]).count("1") != 1]
        checks.append(_chk("V0.4b", "frame strobe is one-hot", not bad_hot, f"frames {bad_hot[:8]}" if bad_hot else ""))
        bad_col = [k for k, f in enumerate(frames) if f[0] >= p.columns]
        checks.append(_chk("V0.4c", f"column index < {p.columns}", not bad_col, f"frames {bad_col[:8]}" if bad_col else ""))
        order_bad = [k for k, f in enumerate(frames)
                     if (f[0], f[1].bit_length() - 1) != (k // p.frames_per_col, k % p.frames_per_col)]
        checks.append(_chk("V0.5a", "frames are in canonical order (column ascending, then frame 0..N-1)", not order_bad and len(frames) == p.expected_frames,
                           f"{len(frames)} frames, expected {p.expected_frames}; first out-of-order frame index: {order_bad[:5]}"))
        seen, dup = {}, []
        for k, f in enumerate(frames):
            key = (f[0], f[1].bit_length() - 1)
            if key in seen:
                dup.append((key, seen[key], k))
            seen[key] = k
        checks.append(_chk("V0.5b", "no (column, frame) is written twice", not dup, f"duplicates: {dup[:4]}" if dup else ""))
        want = {(c, f) for c in range(p.columns) for f in range(p.frames_per_col)}
        miss = sorted(want - set(seen))
        checks.append(_chk("V0.5c", "every (column, frame) of the fabric is written", not miss, f"{len(miss)} missing, e.g. {miss[:4]}", "warning"))
        ones = sum(bin(x).count("1") for f in frames for x in f[2])
        stats.update(frames=len(frames), set_bits=ones, zero_frames=sum(all(x == 0 for x in f[2]) for f in frames),
                     columns_seen=sorted({f[0] for f in frames}))
    # which rows_per_frame values would make this stream parse at all? (independent of the profile)
    valid = []
    for r in range(1, 65):
        try:
            parse_frames(data, p, r); valid.append(r)
        except BitstreamFormatError:
            pass
    stats["rows_per_frame_that_parse"] = valid
    checks.append(_chk("V0.6", "profile rows_per_frame is consistent with the stream", p.rows_per_frame in valid,
                       f"stream parses with rows_per_frame in {valid}; profile says {p.rows_per_frame}"))
    ok = all(c["ok"] for c in checks if c["severity"] == "error")
    return {"stage": "S10/V0", "level": "structural", "ok": ok, "profile": p.name, "checks": checks, "stats": stats,
            "limits": "structural audit cannot see bit flips inside frame data; use audit_against_fasm for that"}


# ----------------------------------------------------------------------------------------- spec + FASM decode
def load_spec(path_or_text, suffix=None):
    """Spec as {tile: {feature: {bit_index(int): '0'|'1'}}} from FABulous bitStreamSpec.csv or the reduced .json fixture."""
    path = Path(path_or_text) if isinstance(path_or_text, (str, Path)) and "\n" not in str(path_or_text) and Path(path_or_text).exists() else None
    text = path.read_text() if path else path_or_text
    suffix = (path.suffix if path else suffix or "").lower()
    if suffix == ".json" or text.lstrip().startswith("{") and not re.match(r"\s*X\d+Y\d+", text):
        raw = json.loads(text)
        return {t: {f: {int(k): v for k, v in d.items()} for f, d in feats.items()} for t, feats in raw.items()}
    import ast
    spec, tile = {}, None
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) == 1 and re.fullmatch(r"X\d+Y\d+", row[0]):
            tile = row[0]; spec[tile] = {}; continue
        if tile and len(row) >= 2:
            spec[tile][row[0]] = {int(k): v for k, v in ast.literal_eval(row[1]).items()}
    return spec


_LIT = re.compile(r"(\d*)'([bhd])([0-9a-fA-F_]+)")   # width optional: nextpnr writes 'b0101 (unsized)


def parse_fasm(text):
    """Return ([(tile, base_name, bit_or_None, value)], unparsed_lines). Value is the full bit pattern as '0'/'1' string for ranges."""
    feats, bad = [], []
    for raw in text.split("\n"):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if "=" in line:
            lhs, rhs = (s.strip() for s in line.split("=", 1))
            m, v = re.fullmatch(r"(X\d+Y\d+)\.(.+)\[(\d+):(\d+)\]", lhs), _LIT.fullmatch(rhs)
            if not (m and v):
                bad.append(raw); continue
            kind, digits = v.group(2), v.group(3).replace("_", "")
            n = int(digits, {"b": 2, "h": 16, "d": 10}[kind])
            hi, lo = int(m.group(3)), int(m.group(4))
            width = int(v.group(1)) if v.group(1) else hi - lo + 1
            feats.append((m.group(1), m.group(2), (hi, lo), format(n, f"0{width}b")[-(hi - lo + 1):]))
        else:
            m = re.fullmatch(r"(X\d+Y\d+)\.(\S+)", line)
            if not m:
                bad.append(raw); continue
            feats.append((m.group(1), m.group(2), None, "1"))
    return feats, bad


def _lookup(spec, tile, name):
    d = spec.get(tile, {})
    if name in d:
        return d[name]
    if name.endswith("[0]") and name[:-3] in d:          # the spec spells bit 0 of a vector without the index
        return d[name[:-3]]
    return None


def required_bits(fasm_text, spec):
    feats, bad = parse_fasm(fasm_text)
    req, unknown = {}, []
    for tile, base, rng, val in feats:
        if rng is None:
            names = [(base, "1")]
        else:
            hi, lo = rng
            names = [(f"{base}[{lo + k}]", ch) for k, ch in enumerate(reversed(val))]
        for name, ch in names:
            if ch != "1":
                continue
            e = _lookup(spec, tile, name)
            if e is None:
                unknown.append(f"{tile}.{name}"); continue
            for idx, v in e.items():
                if v == "1":
                    req.setdefault((tile, idx), []).append(f"{tile}.{name}")
    return req, unknown, bad


def decode_set_bits(data, p):
    frames, _ = parse_frames(data, p)
    out = set()
    top = p.rows_per_frame if p.top_row_y is None else p.top_row_y
    for col, strobe, rows, _h in frames:
        f = strobe.bit_length() - 1
        for r, word in enumerate(rows):
            y = top - r
            for bit in range(p.frame_bits_per_row):
                if (word >> bit) & 1:
                    out.add((f"X{col}Y{y}", f * 32 + bit))
    return out


def audit_against_fasm(data, p, fasm_text, spec, owner_index=None):
    """Compare the bits actually set in the .bin with the bits the FASM requires. Equality is the pass condition."""
    try:
        got = decode_set_bits(data, p)
    except BitstreamFormatError as e:
        return {"stage": "S10/V0", "level": "semantic", "ok": False, "error": f"bitstream does not parse: {e}"}
    req, unknown, bad = required_bits(fasm_text, spec)
    need = set(req)
    owners = owner_index or {}
    if not owners:
        for t, feats in spec.items():
            for name, d in feats.items():
                for idx, v in d.items():
                    owners.setdefault((t, idx), []).append(f"{t}.{name}={v}")
    missing = sorted(need - got)
    extra = sorted(got - need)
    ok = not missing and not extra and not unknown and not bad
    return {"stage": "S10/V0", "level": "semantic", "ok": ok, "required_bits": len(need), "set_bits": len(got),
            "missing": [{"tile": t, "bit": i, "needed_by": req[(t, i)][:3]} for t, i in missing[:20]], "missing_count": len(missing),
            "unexplained": [{"tile": t, "bit": i, "spec_owners": owners.get((t, i), ["(not in the spec provided)"])[:3]} for t, i in extra[:20]],
            "unexplained_count": len(extra), "fasm_features_not_in_spec": unknown[:20], "unparsed_fasm_lines": bad[:10],
            "layout": p.layout}
