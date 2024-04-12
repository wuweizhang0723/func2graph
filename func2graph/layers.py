import torch
from torch import nn, einsum
from torch.nn import Conv1d
from torch import Tensor
from einops import rearrange
from torch.nn import functional as F
import numpy as np
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Residual(nn.Module):
    """residual block"""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, x, *args, **kwargs):
        return x + self._module(x, *args, **kwargs)



class Residual_For_Attention(nn.Module):
    """residual block"""

    def __init__(self, module, prediction_mode=False):
        super().__init__()
        self._module = module
        self.prediction_mode = prediction_mode

    def forward(self, x, *args, **kwargs):
        module_out = self._module(x, *args, **kwargs)

        if self.prediction_mode == True:
            print('2')
            module_out, attn = module_out
            return (x + module_out), attn
        else:
            return x + module_out




# Use sinudial positional encoding from Attention is All You Need.
# Positional Encoding is only added once in the model before attention layer.
class PositionalEncoding(nn.Module):
    def __init__(self, encoding_dim, num=10):
        super(PositionalEncoding, self).__init__()

        self.pos_table = self._get_sinusoid_encoding_table(num, encoding_dim)

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

        return torch.FloatTensor(sinusoid_table).unsqueeze(0).to(device)

    def forward(self, x):
        print(self.pos_table.shape) ################## TODO
        return x + self.pos_table[:, : x.size(1)]





# Attention Layer ----------------------------------------------------------------------------------------------------
#
class Attention(nn.Module):
    def __init__(
        self,
        dim,  # the input and output has shape (batch_size, len, dim) = (b, n, dim)
        *,
        heads=8,
        dim_key=64,    ########### 16 for simulated data
        to_q_layers=0,
        to_k_layers=0,
        dim_value=16,
        dropout=0.0,
        pos_dropout=0.0,
        prediction_mode=False,
        activation = 'none' # 'sigmoid' or 'tanh' or 'softmax' or 'none' or 'cosine_similarity'
    ):
        super().__init__()
        self.activation = activation
        self.scale = dim_key ** -0.5
        self.heads = heads

        # Q, K, V

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_q_fc_layers = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_key * heads, dim_key * heads)
            )
            for layer in range(to_q_layers)
        )
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k_fc_layers = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_key * heads, dim_key * heads)
            )
            for layer in range(to_k_layers)
        )

        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

        # prediction mode

        self.prediction_mode = prediction_mode

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        for layer in self.to_q_fc_layers:
            q = layer(q)
        k = self.to_k(x)
        for layer in self.to_k_fc_layers:
            k = layer(k)
        # v = self.to_v(x)
        v = x.clone()            # identity mapping

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        if self.activation == 'softmax':
            # logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
            logits = einsum("b h i d, b h j d -> b h i j", q, k)
            attn0 = logits.softmax(dim=-1)  # softmax over the last dimension
        elif self.activation == 'sigmoid':
            # logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
            logits = einsum("b h i d, b h j d -> b h i j", q, k)
            attn0 = F.sigmoid(logits)
        elif self.activation == 'tanh':
            # logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
            logits = einsum("b h i d, b h j d -> b h i j", q, k)
            attn0 = F.tanh(logits)
        elif self.activation == 'none':
            # logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
            logits = einsum("b h i d, b h j d -> b h i j", q, k)
            attn0 = logits
        elif self.activation == 'cosine_similarity':
            # q = q + self.rel_content_bias
            logits = torch.zeros(q.shape[0], q.shape[1], q.shape[2], q.shape[2], requires_grad=True).to(device)
            for i in range(q.shape[0]):
                for j in range(q.shape[1]):
                    logits[i][j] = pairwise_cosine_similarity(q[i][j], q[i][j])
            attn0 = logits
            
        attn = self.attn_dropout(attn0)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)   ######### ????？ F.relu(v)
        out = rearrange(out, "b h n d -> b n (h d)")

        if self.prediction_mode == True:
            print('1')
            # return self.to_out(out), attn0
            return out, attn0   ##########################
        else:
            # return self.to_out(out)  # (b, n, dim)
            return out  # (b, n, dim) ##########################
        


# This attention has constrain that W_Q @ W_k^T is 0 on upper triangular part of the matrix,
# which makes the T*T map (W_Q @ W_k^T) to be causal.
#
class Causal_Temporal_Map_Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dropout=0.0,
        prediction_mode=False,
        activation='none', # 'sigmoid' or 'tanh' or 'softmax' or 'none' or 'cosine_similarity'
        causal_temporal_map='lower_triangle',  # 'off_diagonal_1', 'off_diagonal', 'lower_triangle'
        diff=20,
    ):
        super().__init__()
        self.activation = activation
        # self.scale = dim ** -0.5

        self.causal_temporal_map = causal_temporal_map
        self.diff = diff

        # Q, K, V

        self.W_Q_W_KT = nn.Linear(dim, dim, bias=False)

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

        # prediction mode

        self.prediction_mode = prediction_mode

        # make mask, tau=1, [1,0], [2,1], [3,2], [4,3], ..., [dim-1,dim-1-1], [dim,dim-1]
        # tau=2, [2,0], [3,1], [4,2], [5,3], ..., [dim-1,dim-1-2], [dim,dim-2]
        if causal_temporal_map == 'off_diagonal_1' or causal_temporal_map == 'off_diagonal':
            self.mask = torch.zeros(dim, dim, device=device)  ################# TODO
            for i in range(diff, dim):
                j = i - diff
                self.mask[i][j] = 1  ########

    def forward(self, x_e):

        # mask lower triangular part of W_Q_W_KT, mask should take transpose
        # self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data.triu(diagonal=1)

        # mask
        # W_Q_W_KT.weight is (out_features, in_features), so mask should take transpose
        # self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data * self.mask.T

        if self.causal_temporal_map == 'off_diagonal_1':
            self.W_Q_W_KT.weight.data = torch.ones(self.W_Q_W_KT.weight.data.shape[0], self.W_Q_W_KT.weight.data.shape[1], device=device) * self.mask.T
        elif self.causal_temporal_map == 'off_diagonal':
            self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data * self.mask.T
        elif self.causal_temporal_map == 'lower_triangle':
            self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data.triu(diagonal=1)
        

        v = x_e.clone()  # identity mapping

        x_e_ = self.W_Q_W_KT(x_e)  # (b, n, t)

        # x_ = x_ * self.scale

        logits = einsum("b n t, b m t -> b n m", x_e_, x_e)
        attn0 = logits

        attn = self.attn_dropout(attn0)

        out = einsum("b n m, b m t -> b n t", attn, v)

        if self.prediction_mode == True:
            print('1')
            return out, attn0
        else:
            return out
        

# This attention is built on top of previous Causal_Temporal_Map_Attention
# to make E concat to X, instead of addition
# 
class Causal_Temporal_Map_Attention_2(nn.Module):
    def __init__(
        self,
        dim_X,
        dim_E,
        *,
        dropout=0.0,
        prediction_mode=False,
        causal_temporal_map='lower_triangle',  # 'off_diagonal_1', 'off_diagonal', 'lower_triangle'
        diff=20,
    ):
        super().__init__()

        self.causal_temporal_map = causal_temporal_map
        self.diff = diff

        self.relu = nn.ReLU()
        self.W_Q_W_KT = nn.Linear(dim_X, dim_X, bias=False)
        self.Wx_Q_We_KT = nn.Linear(dim_X, dim_E, bias=False)
        self.We_Q_Wx_KT = nn.Linear(dim_E, dim_X, bias=False)
        self.We_Q_We_KT = nn.Linear(dim_E, dim_E, bias=False)

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

        # prediction mode

        self.prediction_mode = prediction_mode

        if causal_temporal_map == 'off_diagonal_1' or causal_temporal_map == 'off_diagonal':
            self.mask = torch.zeros(dim_X, dim_X, device=device)  ################# TODO
            for i in range(diff, dim_X):
                j = i - diff
                self.mask[i][j] = 1  ########

    def forward(self, x, e):
        # mask
        # W_Q_W_KT.weight is (out_features, in_features), so mask should take transpose
        # self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data * self.mask.T
        # self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data.triu(diagonal=1)

        if self.causal_temporal_map == 'off_diagonal_1':
            self.W_Q_W_KT.weight.data = torch.ones(self.W_Q_W_KT.weight.data.shape[0], self.W_Q_W_KT.weight.data.shape[1], device=device) * self.mask.T
        elif self.causal_temporal_map == 'off_diagonal':
            self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data * self.mask.T
        elif self.causal_temporal_map == 'lower_triangle':
            self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data.triu(diagonal=1)

        v = x.clone()  # identity mapping

        self.W_Q_W_KT.weight.data = self.relu(self.W_Q_W_KT.weight.data)
        attn0 = einsum("b n t, b m t -> b n m", self.W_Q_W_KT(x), x)
        e = e.repeat(x.shape[0],1,1)  # (m, e) to (b, m, e)
        attn1 = einsum("b n e, b m e -> b n m", self.Wx_Q_We_KT(x), e)
        attn2 = einsum("b n t, b m t -> b n m", self.We_Q_Wx_KT(e), x)
        attn3 = einsum("b n e, b m e -> b n m", self.We_Q_We_KT(e), e)

        attn = attn0 + attn1 + attn2 + attn3
        attn = self.attn_dropout(attn)

        out = einsum("b n m, b m t -> b n t", attn, v)

        if self.prediction_mode == True:
            return out, attn, attn3
        else:
            return out





# spatial_temporal_1: spatial attention has no value matrix
# spatial_temporal_2: temporal attention has no value matrix
# spatial_temporal_3: spatial attention has value matrix, temporal attention has value matrix
# spatial: there is no temporal attention
#
class Spatial_Temporal_Attention(nn.Module):
    def __init__(
        self,
        dim_T,  # the input and output has shape (batch_size, len, dim) = (b, t, dim_T)
        dim_S,  # the input and output has shape (batch_size, len, dim) = (b, n, dim_S)
        *,
        heads=1,
        dim_key_T=10,
        dim_value_T=10,    # dim_value_T = dim_T = N must be satisfied
        dim_key_S=199,
        dim_value_S=199,   # dim_value_S = dim_S = T must be satisfied
        dropout=0.0,
        pos_dropout=0.0,
        prediction_mode=False,
        attention_type = "spatial_temporal_1",  # "spatial_temporal_1" or "spatial_temporal_2" or "spatial_temporal_3" or "spatial"
        pos_enc_type="none",  # "sin_cos" or "lookup_table" or "none"
    ):
        super().__init__()
        # dim_S = T, dim_T = N

        self.scale_T = dim_key_T ** -0.5
        self.scale_S = dim_key_S ** -0.5
        self.heads = heads

        self.attention_type = attention_type

        # Q_T, K_T, V_T

        self.to_q_T = nn.Linear(dim_T, dim_key_T * heads, bias=False)
        self.to_k_T = nn.Linear(dim_T, dim_key_T * heads, bias=False)
        self.to_v_T = nn.Linear(dim_T, dim_value_T * heads, bias=False)

        # Q_S, K_S, V_S

        self.to_q_S = nn.Linear(dim_S, dim_key_S * heads, bias=False)
        self.to_k_S = nn.Linear(dim_S, dim_key_S * heads, bias=False)
        self.to_v_S = nn.Linear(dim_S, dim_value_S * heads, bias=False)

        self.to_out_T = nn.Linear(dim_value_T * heads, dim_T)
        self.to_out_S = nn.Linear(dim_value_S * heads, dim_S)

        self.to_out_T_S = nn.Linear(dim_S + dim_S, dim_S)

        self.rel_content_bias_T = nn.Parameter(torch.randn(1, heads, 1, dim_key_T))
        self.rel_content_bias_S = nn.Parameter(torch.randn(1, heads, 1, dim_key_S))

        nn.init.zeros_(self.to_out_T.weight)
        nn.init.zeros_(self.to_out_T.bias)
        nn.init.zeros_(self.to_out_S.weight)
        nn.init.zeros_(self.to_out_S.bias)

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

        # prediction mode

        self.prediction_mode = prediction_mode

    def forward(self, x): # x_T: b*T*N, x_S: b*N*T
        x_T, x_S = x
        n, t, h, device = x_S.shape[-2], x_S.shape[-1], self.heads, x_S.device

        q_T = self.to_q_T(x_T)
        k_T = self.to_k_T(x_T)
        v_T = self.to_v_T(x_T)

        q_S = self.to_q_S(x_S)
        k_S = self.to_k_S(x_S)
        v_S = self.to_v_S(x_S)

        q_T, k_T, v_T = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q_T, k_T, v_T))
        q_T = q_T * self.scale_T

        q_S, k_S, v_S = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q_S, k_S, v_S))
        q_S = q_S * self.scale_S

        logits_T = einsum("b h i d, b h j d -> b h i j", q_T + self.rel_content_bias_T, k_T)
        logits_S = einsum("b h i d, b h j d -> b h i j", q_S + self.rel_content_bias_S, k_S)
        # logits_S = torch.abs(logits_S)

        attn_T = logits_T.softmax(dim=-1)  # softmax over the last dimension
        attn_S_0 = logits_S.softmax(dim=-1)  # softmax over the last dimension

        attn_T = self.attn_dropout(attn_T)
        attn_S = self.attn_dropout(attn_S_0)

        out_T = einsum("b h i j, b h j d -> b h i d", attn_T, v_T)
        out_S = einsum("b h i j, b h j d -> b h i d", attn_S, v_S)

        if self.attention_type == "spatial_temporal_1":
            # multiply attn_S with out_T
            out_T = einsum("b h i j, b h t j -> b h t i", attn_S, out_T)
        elif self.attention_type == "spatial_temporal_2":
            # multiply attn_T with out_S
            out_S = einsum("b h i j, b h n j -> b h n i", attn_T, out_S)

        out_T = rearrange(out_T, "b h t d -> b t (h d)")
        out_S = rearrange(out_S, "b h n d -> b n (h d)")

        out_T = self.to_out_T(out_T)  # (b, t, dim_T)
        out_S = self.to_out_S(out_S) # (b, n, dim_S)

        if self.attention_type == "spatial_temporal_3":
            # transpose out_T
            # concate out_S transpose and out_T
            print(out_S.shape)
            print(out_T.shape)
            out_T_S = torch.cat((out_S, out_T.permute(0, 2, 1)), dim=-1)  # (b, dim_T, t+dim_S)
            out_T_S = self.to_out_T_S(out_T_S)  # (b, n=dim_T, t=dim_S)

        
        if self.prediction_mode == True:
            if self.attention_type == "spatial_temporal_1" :
                return out_T.permute(0, 2, 1), attn_S_0
            elif self.attention_type == "spatial_temporal_2" or self.attention_type == "spatial":
                return out_S, attn_S_0
            elif self.attention_type == "spatial_temporal_3":
                return out_T_S, attn_S_0
        else:
            if self.attention_type == "spatial_temporal_1" :
                return out_T.permute(0, 2, 1)
            elif self.attention_type == "spatial_temporal_2" or self.attention_type == "spatial":
                return out_S
            elif self.attention_type == "spatial_temporal_3":
                return out_T_S