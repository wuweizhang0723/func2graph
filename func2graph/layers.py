import torch
from torch import nn, einsum
from torch.nn import Conv1d
from torch import Tensor
from einops import rearrange
from torch.nn import functional as F
import numpy as np
import math
from typing import Union



class Residual(nn.Module):
    """residual block"""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, x, *args, **kwargs):
        return x + self._module(x, *args, **kwargs)




# Use sin-cos positional encoding from Attention is All You Need.
# Positional Encoding is only added once.
class PositionalEncoding(nn.Module):
    def __init__(self, encoding_dim, neuron_num=10):
        super(PositionalEncoding, self).__init__()

        self.pos_table = self._get_sinusoid_encoding_table(neuron_num, encoding_dim)

    def _get_sinusoid_encoding_table(self, n_position, encoding_dim):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / encoding_dim)
                for hid_j in range(encoding_dim)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        print(self.pos_table.shape) ################## TODO
        return x + self.pos_table[:, : x.size(1)].clone().detach()




# Attention Layer ------------------------------------------------------------------------
#
# Positional encoding is added to the input only once before attention layers
class Attention(nn.Module):
    def __init__(
        self,
        dim,  # the input has shape (batch_size, len, dim) = (b, n, dim)
        *,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads
        self.dim = dim

        # Q, K, V

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
        attn = logits.softmax(dim=-1)  # softmax over the last dimension
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)  # (b, n, dim)