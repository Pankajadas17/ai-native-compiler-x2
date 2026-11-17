"""
Connect the pins of a custom user design in the top wrapper FABulous generated. Standard library only (runs inside the Ubuntu toolchain venv).

Why: FABulous 2.2.0 (fabric_cad/gen_design_top_wrapper.py) wires io_in/io_out/io_oeb ONLY for a design literally named `sequential_16bit_en`.
For any other design it prints "Custom design detected, please connect the ports manually" and writes
    <top> user_design_i (.clk(clk), .io_in(), .io_out(), .io_oeb());
Yosys then deletes all user logic as unused: 0 logic cells, empty FASM, all-zero bitstream (a real run showed exactly this).
This tool writes the same connections FABulous itself uses for its demo design:
    .io_in(IO_1_bidirectional_frame_config_pass_O), .io_out(..._I), .io_oeb(..._T)

    python3 tools/fab_connect.py <top_wrapper.v> <user_top_module>
Exit code 0 = connected (or already connected); non-zero = could not connect (the flow must not be run).
"""
import re
import sys
from pathlib import Path

BEL = "IO_1_bidirectional_frame_config_pass"
CONNECTED = f".clk(clk), .io_in({BEL}_O), .io_out({BEL}_I), .io_oeb({BEL}_T)"


def connect(text, top):
    """Return (new_text, message). Raises ValueError with a plain-language reason when it cannot be done safely."""
    if not re.search(rf"\bwire\s*\[\d+:0\]\s*{BEL}_O\b", text):
        raise ValueError(f"the wrapper has no {BEL} pins (wire {BEL}_O not declared): this fabric needs another connection scheme")
    pat = re.compile(rf"^(?P<head>\s*{re.escape(top)}\s+user_design_i\s*)\((?P<args>[^;]*)\)\s*;[ \t]*$", re.M)
    found = list(pat.finditer(text))
    if len(found) != 1:
        raise ValueError(f"expected exactly one '{top} user_design_i (...)' instance in the wrapper, found {len(found)}")
    args = " ".join(found[0].group("args").split())
    if args == CONNECTED:
        return text, "already connected"
    if args.replace(" ", "") != ".clk(clk),.io_in(),.io_out(),.io_oeb()":
        raise ValueError(f"unexpected port list in the generated wrapper: ({args}); refusing to overwrite it")
    new = pat.sub(lambda m: f"{m.group('head')}({CONNECTED});", text, count=1)
    return new, f"connected {top}: io_in <- {BEL}_O, io_out -> {BEL}_I, io_oeb -> {BEL}_T"


def main(argv):
    if len(argv) != 3:
        print(__doc__); return 2
    path = Path(argv[1])
    try:
        new, msg = connect(path.read_text(encoding="utf-8"), argv[2])
    except (ValueError, OSError) as e:
        print(f"fab_connect: CANNOT CONNECT: {e}", file=sys.stderr); return 1
    path.write_text(new, encoding="utf-8")
    print(f"fab_connect: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
