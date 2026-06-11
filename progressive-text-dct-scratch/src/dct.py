"""Differentiable orthonormal DCT-II along the sequence axis, fully vectorized.

D_stack[n] is the n-length DCT matrix zero-padded to (n_max, n_max); a whole
batch of mixed lengths is transformed with one bmm. Gradients flow through
(it's a fixed linear map), so the encoder embedding table is trained
end-to-end through the truncation bottleneck.
"""

import math

import torch


def dct_matrix(n, dtype=torch.float32):
    """Orthonormal DCT-II matrix (n, n); inverse is the transpose."""
    i = torch.arange(n, dtype=torch.float64)
    k = i.unsqueeze(1)
    c = torch.cos(math.pi * (2 * i.unsqueeze(0) + 1) * k / (2 * n))
    scale = torch.full((n, 1), math.sqrt(2.0 / n), dtype=torch.float64)
    scale[0, 0] = math.sqrt(1.0 / n)
    return (scale * c).to(dtype)


class BatchDCT:
    """Batched DCT/IDCT for variable-length sequences padded to n_max."""

    def __init__(self, n_max, device, dtype=torch.float32):
        self.n_max = n_max
        stack = torch.zeros(n_max + 1, n_max, n_max, dtype=dtype)
        for n in range(1, n_max + 1):
            stack[n, :n, :n] = dct_matrix(n, dtype)
        self.stack = stack.to(device)  # (n_max+1, n_max, n_max)

    def forward(self, h, lens):
        """h: (B, n_max, d) zero-padded; lens: (B,) ints. -> (B, n_max, d)
        coefficients (rows >= len are zero)."""
        return torch.bmm(self.stack[lens], h)

    def inverse(self, z, lens):
        """z: (B, n_max, d) coefficients (zero beyond kept ones) -> (B, n_max, d)."""
        return torch.bmm(self.stack[lens].transpose(1, 2), z)
