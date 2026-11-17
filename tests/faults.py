"""Deliberately BROKEN X2 wrappers. Each wraps the real simulator, then corrupts one thing."""
import numpy as np
from emulators.x2_adapter import x2_matmul_report


def _mutate(fn):
    def broken(A, B):
        r = x2_matmul_report(A, B)
        r["output"] = fn(r["output"])
        return r
    return broken

sign_flip = _mutate(lambda o: -o)
off_by_one = _mutate(lambda o: o + 1)
doubled = _mutate(lambda o: o * 2)
transposed_output = _mutate(lambda o: o.T)

def transposed_weights(A, B):          # the exact bug in the old adapter
    return x2_matmul_report(A, np.asarray(B).T)


# ---- mapper bugs (a WRONG tiling plan) used for workload fault injection ----
def drop_last_fold(m):
    m["fold_sizes"] = m["fold_sizes"][:-1]; return m

def drop_last_output_channel(m):
    (k0, k1) = m["k_chunks"][-1]; m["k_chunks"] = m["k_chunks"][:-1] + [(k0, k1 - 1)]; return m


# ---- UNSEEN bugs: deliberately NOT in the knowledge base; the honest diagnosis is "unknown" ----
clipped_output = _mutate(lambda o: np.clip(o, -20, 20))          # saturation of outputs
stale_accumulator = _mutate(lambda o: np.cumsum(o, axis=0))       # each row also carries the previous row's result


# ---- HOLDOUT bugs: created after tuning on the UNSEEN ones. Never tune prompts/rules against these. ----
swap_first_two_columns = _mutate(lambda o: o[:, [1, 0] + list(range(2, o.shape[1]))] if o.shape[1] >= 2 else o)
int8_wraparound = _mutate(lambda o: ((o + 128) % 256) - 128)
zero_last_row = _mutate(lambda o: np.vstack([o[:-1], np.zeros((1, o.shape[1]))]) if o.shape[0] > 1 else o)


# ---- LATENT bugs: invisible to the small-value fixed suite, only triggered by extreme values / long K ----
def latent_wrap16(A, B):                    # 16-bit accumulator wrap: only when |output| >= 32768
    r = x2_matmul_report(A, B); o = r["output"]
    r["output"] = ((o + 32768) % 65536) - 32768
    return r

def latent_long_k(A, B):                    # last row of B ignored when the inner dimension exceeds 8
    B = np.array(B, dtype=float)
    if B.shape[0] > 8:
        B[-1] = 0
    return x2_matmul_report(A, B)

def latent_int8_min_clamp(A, B):            # INT8 minimum -128 silently treated as -127
    return x2_matmul_report(np.maximum(np.asarray(A, float), -127), np.maximum(np.asarray(B, float), -127))
