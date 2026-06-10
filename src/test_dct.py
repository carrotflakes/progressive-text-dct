"""Unit tests: DCT matrix vs scipy, round-trip, and truncation energy order."""

import numpy as np
import scipy.fft
import torch

from model import dct_matrix


def test_dct_matches_scipy():
    for n in [1, 2, 32, 100, 128]:
        x = np.random.RandomState(0).randn(n, 8)
        ref = scipy.fft.dct(x, type=2, norm="ortho", axis=0)
        c = dct_matrix(n, dtype=torch.float64)
        got = (c @ torch.tensor(x)).numpy()
        assert np.allclose(got, ref, atol=1e-10), f"n={n} mismatch"
    print("OK: DCT matches scipy (type 2, norm=ortho)")


def test_roundtrip_and_truncation():
    n, d = 100, 16
    x = torch.randn(n, d, dtype=torch.float64)
    c = dct_matrix(n, dtype=torch.float64)
    z = c @ x
    assert torch.allclose(c.T @ z, x, atol=1e-10), "round trip failed"
    # truncation error must decrease monotonically with K
    errs = []
    for k in [1, 4, 16, 64, 100]:
        zk = z.clone()
        zk[k:] = 0
        errs.append(((c.T @ zk - x) ** 2).mean().item())
    assert all(a >= b for a, b in zip(errs, errs[1:])), "error not monotone"
    assert errs[-1] < 1e-20
    print("OK: orthonormal round-trip and monotone truncation error")


if __name__ == "__main__":
    test_dct_matches_scipy()
    test_roundtrip_and_truncation()
