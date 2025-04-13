import torch
import torch.nn as nn
import toml

# 加载toml配置文件
config = toml.load('configure.toml')


# 使用SISNR(Scale-Invariant Signal-to-Noise Ratio)作为损失函数之一
class loss_sisnr(nn.Module):
    def __init__(self):
        super(loss_sisnr, self).__init__()
        self.window = torch.hann_window(config['FFT']['win_length']).pow(0.5)

    def forward(self, pred_stft, clean_stft):
        # time-frequency representation: (B, F, T, 2)
        eps = 1e-8
        device = pred_stft.device

        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        clean_stft_real, clean_stft_imag = clean_stft[:, :, :, 0], clean_stft[:, :, :, 1]

        wav_ref = torch.istft(pred_stft_real + 1j * pred_stft_imag, **config['FFT'], window=self.window.to(device))
        wav_out = torch.istft(clean_stft_real + 1j * clean_stft_imag, **config['FFT'], window=self.window.to(device))

        # 选取较短的序列长度计算损失
        data_len = min(wav_ref.shape[-1], wav_out.shape[-1])
        wav_ref = wav_ref[..., :data_len]
        wav_out = wav_out[..., :data_len]

        # keepdim=True: 保持维度不变 方便做broadcast计算
        wav_ref = wav_ref - torch.mean(wav_ref, dim=-1, keepdim=True)
        wav_out = wav_out - torch.mean(wav_out, dim=-1, keepdim=True)

        wav_proj = torch.sum(wav_ref * wav_out, dim=-1, keepdim=True) * wav_ref / (torch.sum(wav_ref ** 2,
                                                                                             dim=-1, keepdim=True) + eps)
        wav_vertical = wav_out - wav_proj

        # 计算损失
        sisnr = 10 * torch.log10((torch.sum(wav_proj ** 2, dim=-1) + eps) / (torch.sum(wav_vertical ** 2,
                                                                                       dim=-1) + eps))
        wav_ref_std = torch.std(wav_ref, dim=-1)
        wav_out_std = torch.std(wav_out, dim=-1)

        # 对位比较元素的大小
        com_factor = torch.minimum((wav_ref_std + eps / (wav_out_std + eps)),
                                   (wav_out_std + eps / (wav_ref_std + eps)))

        return -torch.mean(sisnr * com_factor)


# 使用MSE作为损失函数之二
class loss_mse(nn.Module):
    def __init__(self):
        super(loss_mse, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')  # 对所用样本的损失取平均值

    def forward(self, pred_stft, clean_stft):
        # time-frequency representation: (B, F, T, 2)
        eps = 1e-12

        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        clean_stft_real, clean_stft_imag = clean_stft[:, :, :, 0], clean_stft[:, :, :, 1]

        pred_stft_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + eps)
        clean_stft_mag = torch.sqrt(clean_stft_real ** 2 + clean_stft_imag ** 2 + eps)

        pred_stft_real_c = pred_stft_real / (pred_stft_mag ** 0.7)
        pred_stft_imag_c = pred_stft_imag / (pred_stft_mag ** 0.7)

        clean_stft_real_c = clean_stft_real / (clean_stft_mag ** 0.7)
        clean_stft_imag_c = clean_stft_imag / (clean_stft_mag ** 0.7)

        # 计算损失
        loss_real = 30 * self.mse_loss(pred_stft_real_c, clean_stft_real_c)
        loss_imag = 30 * self.mse_loss(pred_stft_imag_c, clean_stft_imag_c)
        loss_mag = 70 * self.mse_loss(pred_stft_mag ** 0.3, clean_stft_mag ** 0.3)

        loss_total = loss_real + loss_imag + loss_mag

        return loss_total


# 总损失函数
class loss_totally(nn.Module):
    def __init__(self):
        super(loss_totally, self).__init__()
        self.loss_sisnr = loss_sisnr()
        self.loss_mse = loss_mse()

    def forward(self, pred_stft, clean_stft):
        pred_stft, clean_stft = pred_stft.permute(0, 2, 3, 1), clean_stft.permute(0, 2, 3, 1)
        loss1 = self.loss_sisnr(pred_stft, clean_stft)
        loss2 = self.loss_mse(pred_stft, clean_stft)
        return loss1 + loss2














