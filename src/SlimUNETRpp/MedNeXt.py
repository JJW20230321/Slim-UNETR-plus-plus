import torch
import torch.nn as nn

class MedNeXtBlock(nn.Module):
    def __init__(self,
                 channels: int  ,
                 #out_channels: int ,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 dim='3d',
                 grn=True
                 ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
           conv = nn.Conv2d
        elif self.dim == '3d':
           conv = nn.Conv3d

        self.conv1 = conv(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels if n_groups is None else n_groups,
        )

        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=channels,
                num_channels=channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=channels,
                data_format='channels_first'
            )

        self.conv2 = conv(
            in_channels=channels,
            out_channels=exp_r * channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.act = nn.GELU()

        self.conv3 = conv(
            in_channels=exp_r * channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * channels, 1, 1, 1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * channels, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1