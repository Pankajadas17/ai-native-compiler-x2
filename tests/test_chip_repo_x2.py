"""$BM_X2_REPO: the X2 behavioral model can come from a separate X2 repo; every reader follows it."""
import pytest

from chip import chip_repo as cr


@pytest.fixture
def repo():
    try:
        return cr.ChipRepo()
    except cr.ChipRepoError:
        pytest.skip("sky130-reram-fpga not cloned here")


def test_default_is_the_chip_repo_copy(repo, monkeypatch):
    monkeypatch.delenv(cr.X2_ENV_VAR, raising=False)
    p = repo.path(cr.INPUT_FILES["x2_behavioral"])
    assert p == (repo.root / cr.INPUT_FILES["x2_behavioral"]).resolve()
    assert cr.x2_model_label() == cr.INPUT_FILES["x2_behavioral"]


def test_file_or_repo_override(repo, monkeypatch, tmp_path):
    f = tmp_path / "x2repo" / cr.X2_MODEL_IN_IP
    f.parent.mkdir(parents=True)
    f.write_text("module sky130_bmlabs_reram__cim32x32(); endmodule\n")
    for value in (str(f), str(tmp_path / "x2repo")):
        monkeypatch.setenv(cr.X2_ENV_VAR, value)
        assert repo.path(cr.INPUT_FILES["x2_behavioral"]) == f.resolve()
        assert repo.read_text(cr.INPUT_FILES["x2_behavioral"]).startswith("module")
        assert cr.x2_model_label().startswith("$BM_X2_REPO")
    assert repo.path(cr.INPUT_FILES["x2_primitive"]).is_relative_to(repo.root)       # other files still from the chip repo


def test_bad_override_is_an_error(repo, monkeypatch, tmp_path):
    monkeypatch.setenv(cr.X2_ENV_VAR, str(tmp_path))
    with pytest.raises(cr.ChipRepoError, match="BM_X2_REPO"):
        repo.path(cr.INPUT_FILES["x2_behavioral"])
