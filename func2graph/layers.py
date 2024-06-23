import torch
from torch import nn, einsum
from einops import rearrange
from torch.nn import functional as F
from torchmetrics.functional.pairwise import pairwise_cosine_similarity



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
            module_out, attn = module_out
            return (x + module_out), attn
        else:
            return x + module_out




############################################################################################################
# Attention Layer
############################################################################################################

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
        


############################################################################################################
# Causal_Temporal_Map_Attention Layer
#
# This attention has constrain that W_Q @ W_k^T is 0 on upper triangular part of the matrix,
# which makes the T*T map (W_Q @ W_k^T) to be causal.
############################################################################################################

class Causal_Temporal_Map_Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dropout=0.0,
        prediction_mode=False,
        activation='none', # 'sigmoid' or 'tanh' or 'softmax' or 'none'
        causal_temporal_map='lower_triangle',  # 'off_diagonal_1', 'off_diagonal', 'lower_triangle'
        diff=20,
    ):
        super().__init__()
        self.activation = activation
        self.scale = dim ** -0.5

        self.layer_norm = nn.LayerNorm(dim)

        self.causal_temporal_map = causal_temporal_map
        self.diff = diff

        # TT

        self.W_Q_W_KT = nn.Linear(dim, dim, bias=False)

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

        # prediction mode

        self.prediction_mode = prediction_mode

        # Make mask
        # e.g. tau=1, [1,0], [2,1], [3,2], [4,3], ..., [dim-1,dim-1-1], [dim,dim-1]
        # e.g. tau=2, [2,0], [3,1], [4,2], [5,3], ..., [dim-1,dim-1-2], [dim,dim-2]
        if causal_temporal_map == causal_temporal_map == 'off_diagonal':
            self.mask = torch.zeros(dim, dim)
            for i in range(diff, dim):
                j = i - diff
                self.mask[i][j] = 1

            # make self.mask be in the same device as self.W_Q_W_KT
            self.mask = self.mask.to(self.W_Q_W_KT.weight.device)

    def forward(self, x, e):

        # Mask

        # W_Q_W_KT.weight is (out_features, in_features), so mask should take transpose
        # self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data * self.mask.T

        if self.causal_temporal_map == 'off_diagonal':
            self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data * self.mask.T
        elif self.causal_temporal_map == 'lower_triangle':
            self.W_Q_W_KT.weight.data = self.W_Q_W_KT.weight.data.triu(diagonal=1)
        
        x_e = x + e
        x_e = self.layer_norm(x_e)
        v = x_e.clone()  # V = X + E ###############################

        # x = self.layer_norm(x)
        # v = x.clone()  # V = X ###################################

        x_e_ = self.W_Q_W_KT(x_e)  # (b, n, t)
        x_e_ = x_e_ * self.scale

        logits = einsum("b n t, b m t -> b n m", x_e_, x_e)
        if self.activation == 'softmax':
            attn0 = logits.softmax(dim=-1)
        elif self.activation == 'sigmoid':
            attn0 = F.sigmoid(logits)
        elif self.activation == 'tanh':
            attn0 = F.tanh(logits)
        elif self.activation == 'none':
            attn0 = logits

        attn = self.attn_dropout(attn0)

        out = einsum("b n m, b m t -> b n t", attn, v)

        e_repeat = e.repeat(x.shape[0],1,1)  # (m, e) to (b, m, e)
        attn3 = einsum("b n e, b m e -> b n m", self.W_Q_W_KT(e_repeat), e_repeat)

        if self.prediction_mode == True:
            return out, attn0, attn3
        else:
            return out
        


############################################################################################################
# Causal_Temporal_Map_Attention_2 Layer
#
# This attention is built on top of previous Causal_Temporal_Map_Attention 
# to make E concat to X, instead of addition.
############################################################################################################

class Causal_Temporal_Map_Attention_2(nn.Module):
    def __init__(
        self,
        dim_X,
        dim_E,
        *,
        dropout=0.0,
        prediction_mode=False,
        causal_temporal_map='lower_triangle',  # 'off_diagonal', 'lower_triangle'
        diff=20,
    ):
        super().__init__()

        # self.layer_norm = nn.LayerNorm(dim_X + dim_E)

        self.causal_temporal_map = causal_temporal_map
        self.diff = diff
        self.scale = (dim_X + dim_E) ** -0.5

        # Q, K

        self.query_linear = nn.Linear(dim_X + dim_E, dim_X + dim_E, bias=False)
        self.key_linear = nn.Linear(dim_X + dim_E, dim_X + dim_E, bias=False)

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

        # prediction mode

        self.prediction_mode = prediction_mode

    def forward(self, x, e):

        e = e.repeat(x.shape[0],1,1)  # (m, e) to (b, m, e)
        x_e = torch.cat((x, e), dim=-1)

        batch_size, n, t = x.shape

        # We_Q_We_KT: (dim_E, dim_E)

        We_Q_We_KT = (self.query_linear.weight.clone().detach().T)[t:] @ (self.key_linear.weight.clone().detach().T)[t:].T
        attn3 = einsum("b n e, b m e -> b n m", e @ We_Q_We_KT, e)

        # Q, K

        queries = self.query_linear(x_e)
        keys = self.key_linear(x_e)

        attn = einsum("b n d, b m d -> b n m", queries, keys)
        attn = self.attn_dropout(attn)
        attn = attn * self.scale

        v = x  # identity mapping
        out = einsum("b n m, b m t -> b n t", attn, v)

        if self.prediction_mode == True:
            return out, attn, attn3
        else:
            return out
