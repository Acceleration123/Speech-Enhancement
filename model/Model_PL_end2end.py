import torch
import torch.nn as nn
import Module
import toml
from ptflops import get_model_complexity_info

config = toml.load('configure.toml')


class stft_encoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        super(stft_encoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(win_length).pow(0.5)

    def forward(self, x):
        # x: (B, T)
        device = x.device
        x_stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                            window=self.window.to(device), return_complex=False)

        return x_stft  # (B, F, T, 2)


class stft_decoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        super(stft_decoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(win_length).pow(0.5)

    def forward(self, x_stft):
        # x_stft: (B, 2, T, F)
        device = x_stft.device
        x_stft = x_stft.permute(0, 3, 2, 1)  # (B, F, T, 2)
        X = torch.complex(x_stft[..., 0], x_stft[..., 1])
        x = torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(device))

        return x  # (B, T)


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


class PL_TF_Grid(nn.Module):
    def __init__(self):
        super(PL_TF_Grid, self).__init__()
        self.num_tf_blks = config['DL_TF-Grid']['num_blocks']
        self.num_middles = config['PL_TF-Grid']['num_middles']

        self.stft_encoder = stft_encoder(**config['FFT'])

        self.phase_encoder = Phase_Encoder(channels=config['Phase_Encoder']['channels'],
                                           kernel_size=config['Phase_Encoder']['kernel_size'],
                                           stride=config['Phase_Encoder']['stride'],
                                           padding=config['Phase_Encoder']['padding'])

        self.conv = nn.Conv2d(in_channels=config['Phase_Encoder']['channels'],
                              out_channels=config['TF-Grid_Block']['embedding'],
                              kernel_size=config['DL_TF-Grid']['kernel_size'], stride=config['DL_TF-Grid']['stride'],
                              padding=config['DL_TF-Grid']['padding'])

        self.gln = nn.GroupNorm(num_groups=1, num_channels=config['TF-Grid_Block']['embedding'])

        self.de_conv = nn.ModuleList([])
        for i in range(self.num_middles + 1):
            self.de_conv.append(
                nn.ConvTranspose2d(in_channels=config['TF-Grid_Block']['embedding'],
                                   out_channels=2, kernel_size=config['DL_TF-Grid']['kernel_size'],
                                   stride=config['DL_TF-Grid']['stride'], padding=config['DL_TF-Grid']['padding'])
            )

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

        self.stft_decoder = stft_decoder(**config['FFT'])

    def forward(self, x):
        """
        x: (B, T)  time domain
        """
        n_samples = x.shape[1]

        y = None
        cumulative_x = None
        x = self.stft_encoder(x)  # (B, F, T, 2) / (B, 257, T, 2)
        x = x.permute(0, 3, 2, 1)  # (B, 2, T, F)
        x = self.phase_encoder(x)  # (B, 4, T, F)
        x = self.conv(x)  # (B, 32, T, F)
        x = self.gln(x)  # (B, 32, T, F)

        # TF-Grid Blocks
        for i in range(self.num_tf_blks):
            cumulative_x = x if cumulative_x is None else cumulative_x + x  # (B, 32, T, F)
            x = self.tf_grid_blks[i](cumulative_x)  # (B, 32, T, F)
            middle = self.de_conv[i](x)  # (B, 2, T, F)
            if y is None:
                y = self.stft_decoder(middle)  # (B, T)
                y = nn.functional.pad(y, (0, n_samples - y.shape[-1]))  # (B, T)
                y = y.unsqueeze(-1)  # (B, T, 1)
            else:
                middle = self.stft_decoder(middle)  # (B, T)
                middle = nn.functional.pad(middle, (0, n_samples - middle.shape[-1]))  # (B, T)
                middle = middle.unsqueeze(-1)  # (B, T, 1)
                y = torch.cat([y, middle], dim=-1)  # (B, T, num_results)

        return y  # (B, T, num_tf_blks) /   (B, T, 5)


if __name__ == '__main__':
    model = PL_TF_Grid().eval()

    y_in = torch.randn(1, 16000)
    y_out = model(y_in)
    print(y_out.shape)

    flops, params = get_model_complexity_info(PL_TF_Grid().eval(), (16000,), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print(flops, params)
