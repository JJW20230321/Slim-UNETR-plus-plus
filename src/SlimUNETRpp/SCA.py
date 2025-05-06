import torch.nn as nn
import torch
import math

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class ChannelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=False, channel_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvv = nn.Linear(hidden_size, hidden_size * 4,
                              bias=qkv_bias)

        self.attn_drop = nn.Dropout(channel_attn_drop)

    def forward(self, x):
        B, C, W, H, Z = x.shape
        x = x.reshape(B, C, W * H * Z).permute(0, 2, 1)
        B, N, C = x.shape

        qkvv = self.qkvv(x)
        qkvv = qkvv.reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        x_CA = x_CA.reshape(B, W, H, Z, C).permute(0, 4, 1, 2, 3)
        return x_CA

class SpatialAttention(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads, qkv_bias=False, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvv = nn.Linear(hidden_size, hidden_size * 4,
                              bias=qkv_bias)

        self.EF = nn.Parameter(
            init_(torch.zeros(input_size, proj_size)))

        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward(self, x):
        B, C, W, H, Z = x.shape

        x = x.reshape(B, C, W * H * Z).permute(0, 2, 1)
        B, N, C = x.shape
        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads,C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)

        v_SA = v_SA.transpose(-2, -1)

        proj_e_f = lambda args: torch.einsum('bhdn,nk->bhdk', *args)

        k_shared_projected, v_SA_projected = map(proj_e_f, zip((k_shared, v_SA), (self.EF, self.EF)))

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        x_SA = x_SA.reshape(B, W, H, Z, C).permute(0, 4, 1, 2, 3)

        return x_SA

def norm(planes, mode='bn', groups=16):
    if mode == 'bn':
        return nn.BatchNorm3d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential

class DANetHead(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads):
        super(DANetHead, self).__init__()

        self.sa = SpatialAttention(input_size, hidden_size, proj_size, num_heads)
        self.sc = ChannelAttention(hidden_size, num_heads)

    def forward(self, x):
        sa_feat = self.sa(x)
        sc_feat = self.sc(x)
        feat_sum = sc_feat + sa_feat
        return feat_sum


