"""Sanity stage 1 (reversibility) + pipeline unit tests.

Run: python src/test_pipeline.py
"""

import numpy as np
import scipy.fft
import torch
import yaml

from dct import BatchDCT, dct_matrix
from model import ScratchLM


def test_dct_scipy():
    for n in [1, 2, 32, 100, 128]:
        x = np.random.RandomState(0).randn(n, 8)
        ref = scipy.fft.dct(x, type=2, norm="ortho", axis=0)
        got = (dct_matrix(n, torch.float64) @ torch.tensor(x)).numpy()
        assert np.allclose(got, ref, atol=1e-10), f"n={n}"
    print("OK: DCT matches scipy (type-2, ortho)")


def test_reversibility():
    """Spec sanity #1: DCT -> keep ALL coefficients -> inverse == identity."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bd = BatchDCT(128, dev, torch.float64)
    B, d = 16, 32
    lens = torch.randint(32, 129, (B,), device=dev)
    h = torch.randn(B, 128, d, device=dev, dtype=torch.float64)
    h = h * (torch.arange(128, device=dev)[None, :, None] < lens[:, None, None])
    z = bd.forward(h, lens)
    h_rec = bd.inverse(z, lens)
    err = (h - h_rec).abs().max().item()
    assert err < 1e-9, f"reversibility error {err}"
    print(f"OK: full-K DCT round trip (max err {err:.2e})")


def test_truncation_monotone():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bd = BatchDCT(128, dev, torch.float64)
    lens = torch.full((4,), 100, device=dev)
    h = torch.randn(4, 128, 16, device=dev, dtype=torch.float64)
    h[:, 100:] = 0
    z = bd.forward(h, lens)
    errs = []
    for k in [1, 4, 16, 64, 100]:
        zk = z.clone()
        zk[:, k:] = 0
        errs.append(((bd.inverse(zk, lens) - h) ** 2).mean().item())
    assert all(a >= b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-18
    print("OK: truncation error monotone in K")


def test_model_grads_and_shapes():
    """Loss backward must reach the ENCODER embedding (end-to-end claim)."""
    dev = "cuda"
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["model"]["n_layers"] = 2  # small for test
    m = ScratchLM(cfg, dev).to(dev)
    B, n_max = 4, cfg["data"]["n_max"]
    ids = torch.randint(0, 16384, (B, n_max), device=dev)
    lens = torch.tensor([40, 128, 32, 77], device=dev)
    k = torch.tensor([5, 64, 1, 13], device=dev)
    k_pad = 64
    idx = torch.zeros(B, k_pad, dtype=torch.long, device=dev)
    valid = torch.zeros(B, k_pad, dtype=torch.bool, device=dev)
    for j in range(B):
        idx[j, : k[j]] = torch.arange(k[j])
        valid[j, : k[j]] = True
    loss = m(ids, lens, idx, valid, mode="dct")
    loss.backward()
    g = m.enc_emb.weight.grad
    assert g is not None and g.abs().sum() > 0, "no grad reached encoder E!"
    print(f"OK: loss={loss.item():.3f}, grad reaches encoder E "
          f"(|g|={g.abs().sum().item():.2e})")
    # generation shape & determinism
    m.eval()
    with torch.no_grad():
        out = m.generate(ids, lens, idx, valid, mode="dct")
    assert out.shape == (B, 128)
    assert (out[2, 32:] == 0).all(), "tokens beyond n must be zeroed"
    print("OK: generate shapes / length masking")


def test_prefix_mask():
    mask = ScratchLM.prefix_lm_mask(
        3, 2, torch.tensor([[True, True, False]]), torch.tensor([[True, True]]))
    m = mask[0, 0]
    # L = 1+3+1+2 = 7: [len, z0, z1, zPAD, bos, t1, t2]
    assert m[1, 2] and m[2, 1], "prefix must be bidirectional"
    assert not m[1, 4], "prefix must not see BOS/text"
    assert not m[0, 3], "padded z slot must be masked as key"
    assert m[5, 0] and m[5, 4], "text sees prefix and BOS"
    assert not m[5, 6] and m[6, 5], "text is causal"
    print("OK: prefix-LM attention mask semantics")


def test_task3_archs():
    """A (enc+DCT) and B (enc+latents): grads must reach encoder blocks,
    E, and latent queries; generation shapes hold."""
    dev = "cuda"
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["model"]["n_layers"] = 2
    cfg["encoder"]["n_layers"] = 2
    B, n_max = 4, cfg["data"]["n_max"]
    ids = torch.randint(0, 16384, (B, 100), device=dev)  # n_pad < n_max on purpose
    lens = torch.tensor([40, 100, 32, 77], device=dev)
    k_pad = 64
    idx = torch.zeros(B, k_pad, dtype=torch.long, device=dev)
    valid = torch.zeros(B, k_pad, dtype=torch.bool, device=dev)
    for j, kj in enumerate([5, 64, 1, 13]):
        kj = min(kj, int(lens[j]))
        idx[j, :kj] = torch.arange(kj, device=dev)
        valid[j, :kj] = True

    for arch, mode in ((dict(encoder="transformer", bottleneck="dct"), "dct"),
                       (dict(encoder="transformer", bottleneck="latent"),
                        "latent")):
        m = ScratchLM(cfg, dev, **arch).to(dev)
        loss = m(ids, lens, idx, valid, mode=mode)
        loss.backward()
        assert m.enc_emb.weight.grad.abs().sum() > 0, f"{arch}: no grad to E"
        g_enc = sum(p.grad.abs().sum().item()
                    for p in m.enc_blocks.parameters())
        assert g_enc > 0, f"{arch}: no grad to encoder blocks"
        if mode == "latent":
            assert m.latent_q.grad.abs().sum() > 0, "no grad to latent queries"
        m.eval()
        with torch.no_grad():
            out = m.generate(ids, lens, idx, valid, mode=mode)
        assert out.shape == (B, 100)
        assert (out[2, 32:] == 0).all()
        print(f"OK: task3 arch {arch['bottleneck']} loss={loss.item():.3f}, "
              f"grads reach E/encoder{'/latents' if mode == 'latent' else ''}")


if __name__ == "__main__":
    test_dct_scipy()
    test_reversibility()
    test_truncation_monotone()
    test_prefix_mask()
    test_model_grads_and_shapes()
    test_task3_archs()
    print("ALL TESTS PASSED")
