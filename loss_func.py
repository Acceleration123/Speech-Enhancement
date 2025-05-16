import torch
import torch.nn as nn


class loss_stft(nn.Module):
    def __init__(
            self,
            n_fft=512,
            hop_length=256,
            window=torch.hann_window(512).pow(0.5),
            compress_factor=0.7,
            lamda_ri=30,
            lamda_mag=70,
            eps=1e-8
    ):
        super(loss_stft, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.c = compress_factor
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag
        self.eps = eps
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, clean):
        """
        pred: (B, T)
        clean: (B, T)
        """
        device = pred.device
        pred_tf = torch.stft(pred, n_fft=self.n_fft, hop_length=self.hop_length,
                             window=self.window.to(device), return_complex=True)  # (B, F, T)
        clean_tf = torch.stft(clean, n_fft=self.n_fft, hop_length=self.hop_length,
                              window=self.window.to(device), return_complex=True)  # (B, F, T)

        pred_tf_mag_c = torch.abs(pred_tf).clamp(self.eps).pow(self.c)   # (B, F, T)
        clean_tf_mag_c = torch.abs(clean_tf).clamp(self.eps).pow(self.c)  # (B, F, T)

        # magnitude compressed
        pred_tf_c = pred_tf / pred_tf_mag_c  # (B, F, T)
        clean_tf_c = clean_tf / clean_tf_mag_c  # (B, F, T)

        loss_mag = self.mse(pred_tf_mag_c, clean_tf_mag_c)
        loss_real = self.mse(pred_tf_c.real, clean_tf_c.real)
        loss_imag = self.mse(pred_tf_c.imag, clean_tf_c.imag)

        return self.lamda_mag * loss_mag + self.lamda_ri * (loss_real + loss_imag)


class loss_sisnr(nn.Module):
    def __init__(self, eps=1e-8):
        super(loss_sisnr, self).__init__()
        self.eps = eps

    def forward(self, pred, clean):
        """
        pred: (B, T)
        clean: (B, T)
        """
        norm = (torch.sum(clean * pred, dim=-1, keepdim=True) * clean /
                  (torch.sum(torch.square(clean), dim=-1, keepdim=True) + self.eps))

        sisnr = -2 * torch.log10(torch.norm(norm, dim=-1, keepdim=True) /
                                 torch.norm(pred - norm, dim=-1, keepdim=True).clamp(self.eps) + self.eps).mean()
        return sisnr


class loss_hybrid(nn.Module):
    def __init__(self, pl=False):
        super(loss_hybrid, self).__init__()
        self.pl = pl
        self.loss_stft = loss_stft()
        self.loss_sisnr = loss_sisnr()

    def forward(self, pred, label):
        """
        pred: (B, T) in direct learning mode, (B, T, 5) in progressive learning mode
        label: (B, T) in direct learning mode, (B, T, 5) in progressive learning mode
        """
        if self.pl:
            num = label.shape[-1]
            loss_tot = 0
            for i in range(num):
                loss_tot += (self.loss_stft(pred[:, :, i], label[:, :, i]) +
                             self.loss_sisnr(pred[:, :, i], label[:, :, i]))
        else:
            loss_tot = self.loss_stft(pred, label) + self.loss_sisnr(pred, label)
        return loss_tot


if __name__ == '__main__':
    # test
    x_dl = torch.randn(1, 16000)
    y_dl = torch.randn(1, 16000)
    loss_dl = loss_hybrid(pl=False)
    print(loss_dl(x_dl, y_dl))

    x_pl = torch.randn(1, 16000, 5)
    y_pl = torch.randn(1, 16000, 5)
    loss_pl = loss_hybrid(pl=True)
    print(loss_pl(x_pl, y_pl))

