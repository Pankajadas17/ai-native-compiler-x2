"""x2c CLI: every command maps to a module that exists, arguments pass through, Linux-only commands refuse elsewhere."""
import importlib.util

import pytest

from x2c import cli


def test_every_command_points_at_an_existing_module():
    missing = [f"{n} -> {m}" for n, (m, _, _) in cli.COMMANDS.items()
               if not m.startswith("@") and importlib.util.find_spec(f"x2c.{m}") is None]
    assert not missing, missing


def test_recheck_only_uses_known_commands():
    assert all(name in cli.COMMANDS for name, _ in cli.RECHECK)


def test_list_and_unknown(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "deploy" in out and "sim engine2" in out and "[Linux]" in out
    assert cli.main(["no-such-command"]) == 2
    assert cli.main(["sim", "no-such-part"]) == 2


def test_group_alone_lists_that_group(capsys):
    assert cli.main(["sim"]) == 0
    out = capsys.readouterr().out
    assert "sim engine2" in out and "\n  deploy " not in out


def test_linux_only_commands_refuse_elsewhere(monkeypatch, capsys):
    monkeypatch.setattr(cli, "on_linux", lambda: False)
    assert cli.main(["build", "--example", "x2_engine2_4bit"]) == 2
    assert "Ubuntu terminal" in capsys.readouterr().out


@pytest.mark.parametrize("argv, module, rest", [
    (["deploy", "--workload", "mnist"], "x2c.commands.deploy", ["--workload", "mnist"]),
    (["sim", "engine2", "--sim"], "x2c.commands.sim_engine2", ["--sim"]),
    (["chip", "costs"], "x2c.chip.fabric_costs", []),
])
def test_arguments_pass_through(monkeypatch, argv, module, rest):
    seen = {}
    monkeypatch.setattr(cli, "on_linux", lambda: True)

    def fake_run_module(name, run_name):
        seen["module"], seen["argv"] = name, list(cli.sys.argv)
        raise SystemExit(0)
    monkeypatch.setattr(cli.runpy, "run_module", fake_run_module)
    assert cli.main(argv) == 0
    assert seen["module"] == module and seen["argv"][1:] == rest


def test_exit_codes(monkeypatch):
    monkeypatch.setattr(cli, "on_linux", lambda: True)
    for code, want in ((None, 0), (0, 0), (3, 3), ("message", 1)):
        def fake(name, run_name, code=code):
            raise SystemExit(code)
        monkeypatch.setattr(cli.runpy, "run_module", fake)
        assert cli.main(["replay", "deploy/mnist"]) == want
