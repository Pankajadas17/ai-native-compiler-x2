import BMsemi_Model.bm_models.core.pe as pe
from x2c.bmsemi.x2_adapter import x2_matmul_report
import numpy as np

def test_real_simulator_is_used():
    assert "BMsemi_Model" in pe.__file__
    r = x2_matmul_report(np.array([[1, 2]]), np.array([[5, 6], [7, 8]]))
    assert r["backend"] == "behavioral_simulator" and r["physical_hardware"] is False
    assert r["total_cycles"] > 100          # a NumPy stand-in would be tiny

def test_orientation_non_square():
    from x2c.bmsemi.x2_adapter import x2_matmul
    A = np.arange(6).reshape(2, 3); B = np.arange(12).reshape(3, 4)
    assert np.allclose(x2_matmul(A, B)[0], A @ B)
