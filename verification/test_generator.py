import numpy as np
def make_cases(seed=0, n_random=20):
    rng = np.random.default_rng(seed)
    cases = [("fixed_2x2", np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])),
             ("zeros", np.zeros((3,3)), np.ones((3,3))),
             ("identity", np.eye(4), np.arange(16).reshape(4,4)),
             ("negatives", np.array([[-1,2],[3,-4]]), np.array([[5,-6],[-7,8]])),
             ("non_square", np.arange(6).reshape(2,3), np.arange(12).reshape(3,4))]
    for i in range(n_random):
        m, k, n = rng.integers(1, 9, 3)
        cases.append((f"random_{i}", rng.integers(-8, 9, (m,k)), rng.integers(-8, 9, (k,n))))
    return cases
