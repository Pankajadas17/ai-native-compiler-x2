import numpy as np
from x2c.bmsemi.x2_adapter import x2_matmul
from x2c.verification.matmul_check import golden_matmul
from x2c.verification.matmul_check import compare
from x2c.commands.bmsemi_verify import run_all

def test_fixed_2x2():
    A=np.array([[1,2],[3,4]]); B=np.array([[5,6],[7,8]])
    assert golden_matmul(A,B).tolist()==[[19,22],[43,50]]
    assert compare(golden_matmul(A,B), x2_matmul(A,B)[0])["passed"]

def test_comparator_catches_mismatch():
    assert not compare([[1,2]], [[1,3]])["passed"]

def test_full_suite():
    s = run_all(); assert s["passed"] == s["total"]
