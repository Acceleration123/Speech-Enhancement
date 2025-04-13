import torch
import torch.nn as nn

import Module
import toml
from ptflops import get_model_complexity_info

config = toml.load('configure.toml')


class Phase_Encoder(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(Phase_Encoder, self).__init__()

        self.conv_real = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding)

    def forward(self, x):
        # x: (B, C, T, F) where C = 2
        x_real, x_imag = x[:, 0, :, :], x[:, 1, :, :]  # (B, T, F)
        x_real = x_real.unsqueeze(1)  # (B, 1, T, F)
        x_imag = x_imag.unsqueeze(1)  # (B, 1, T, F)
        x_conv_real = self.conv_real(x_real) - self.conv_imag(x_imag)  # (B, C, T, F)
        x_conv_imag = self.conv_real(x_imag) + self.conv_imag(x_real)  # (B, C, T, F)
        x_conv = x_conv_real + 1j * x_conv_imag  # (B, C, T, F)
        # Dynamic Compression
        x_dc = torch.abs(x_conv).pow(0.5)  # (B, C, T, F)

        return x_dc  # (B, C, T, F)


class DL_TF_Grid(nn.Module):
    def __init__(self):
        super(DL_TF_Grid, self).__init__()
        self.num_tf_blks = config['DL_TF-Grid']['num_blocks']
        self.embedding = config['TF-Grid_Block']['embedding']
        self.hidden_size = config['TF-Grid_Block']['hidden_size']
        self.kernel_size = config['TF-Grid_Block']['kernel_size']
        self.stride = config['TF-Grid_Block']['stride']

        self.phase_encoder = Phase_Encoder(channels=config['Phase_Encoder']['channels'],
                                           kernel_size=config['Phase_Encoder']['kernel_size'],
                                           stride=config['Phase_Encoder']['stride'],
                                           padding=config['Phase_Encoder']['padding'])

        self.conv = nn.Conv2d(in_channels=config['Phase_Encoder']['channels'],
                              out_channels=config['TF-Grid_Block']['embedding'],
                              kernel_size=config['DL_TF-Grid']['kernel_size'], stride=config['DL_TF-Grid']['stride'],
                              padding=config['DL_TF-Grid']['padding'])

        self.gln = nn.GroupNorm(num_groups=1, num_channels=config['TF-Grid_Block']['embedding'])

        self.tf_grid_blks = nn.ModuleList([])
        for i in range(self.num_tf_blks):
            self.tf_grid_blks.append(Module.GridNetBlock(
                emb_dim=config['TF-Grid_Block']['embedding'],
                emb_ks=config['TF-Grid_Block']['kernel_size'],
                emb_hs=config['TF-Grid_Block']['stride'],
                n_freqs=config['TF-Grid_Block']['width'],
                hidden_channels=config['TF-Grid_Block']['hidden_size'],
                n_head=config['FSM']['num_heads']
            ))

        self.de_conv = nn.ConvTranspose2d(in_channels=config['TF-Grid_Block']['embedding'],
                                          out_channels=2, kernel_size=config['DL_TF-Grid']['kernel_size'],
                                          stride=config['DL_TF-Grid']['stride'],
                                          padding=config['DL_TF-Grid']['padding'])

    def forward(self, x):
        # x: (B, F, T, 2) / (B, 257, T, 2)
        x = x.permute(0, 3, 2, 1)  # (B, 2, T, F) / (B, 2, T, 257)
        x = self.phase_encoder(x)  # (B, 4, T, F) / (B, 4, T, 257)
        x = self.conv(x)  # (B, 32, T, F) / (B, 32, T, 255)
        x = self.gln(x)  # (B, 32, T, F) / (B, 32, T, 255)

        # TF-Grid Blocks
        for i in range(self.num_tf_blks):
            x = self.tf_grid_blks[i](x)  # (B, 32, T, F) / (B, 32, T, 255)

        x = self.de_conv(x)  # (B, 2, T, F) / (B, 2, T, 257)
        return x.permute(0, 1, 3, 2)   # (B, 2, F, T)


if __name__ == '__main__':
    flops, params = get_model_complexity_info(DL_TF_Grid().eval(), (257, 20, 2), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print(flops, params)
