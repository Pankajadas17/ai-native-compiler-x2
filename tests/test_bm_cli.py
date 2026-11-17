"""bm.py: one entry point; every command maps to a script that exists, and Ubuntu-only commands refuse on Windows."""
import bm


def test_every_command_points_at_an_existing_script():
    missing = [f"{n} -> {s}" for n, (s, _, _) in bm.COMMANDS.items() if isinstance(s, str) and not s.startswith("@")
               and not (bm.ROOT / s).is_file()]
    assert not missing, missing


def test_workflows_only_use_known_commands():
    for steps in (bm.WINDOWS_ALL, bm.UBUNTU_ALL, bm.X2_RECHECK):
        assert all(c in bm.COMMANDS for c, _ in steps)


def test_list_and_unknown(capsys):
    assert bm.main([]) == 0 and "deploy" in capsys.readouterr().out
    assert bm.main(["no-such-command"]) == 2


def test_ubuntu_only_commands_refuse_elsewhere(monkeypatch, capsys):
    monkeypatch.setattr(bm, "on_ubuntu", lambda: False)
    assert bm.run("build", ["--example", "x2_engine2_4bit"]) == 2
    assert "Ubuntu terminal" in capsys.readouterr().out


def test_arguments_pass_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(bm.subprocess, "run", lambda cmd, cwd: seen.setdefault("cmd", cmd) and type("R", (), {"returncode": 0})())
    assert bm.run("deploy", ["--workload", "mnist"]) == 0
    assert seen["cmd"][1].endswith("deploy_chip.py") and seen["cmd"][2:] == ["--workload", "mnist"]
