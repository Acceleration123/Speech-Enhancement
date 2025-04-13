import torch
import os
import soundfile as sf
from torch.utils.data import Dataset


class TrainDataset_DL(Dataset):
    def __init__(self, noisy_dir, clean_dir, window_length, hop_length, n_fft):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        self.noisy_list = os.listdir(self.noisy_dir)
        self.clean_list = os.listdir(self.clean_dir)

        self.window = torch.hann_window(window_length=window_length).pow(0.5)
        self.hop_length = hop_length
        self.n_fft = n_fft

    def __len__(self):
        return len(self.noisy_list)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_list[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_list[idx])

        noisy, fs = sf.read(noisy_path, dtype='float32')
        clean, _ = sf.read(clean_path, dtype='float32')

        wav_len = int(min(len(noisy), len(clean)) // 3)
        noisy = noisy[:wav_len]
        clean = clean[:wav_len]

        # 记得用torch.tensor()把数据转成tensor
        noisy = torch.tensor(noisy)
        clean = torch.tensor(clean)

        noisy_stft = torch.stft(noisy, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window, return_complex=True)
        clean_stft = torch.stft(clean, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window, return_complex=True)

        x_input = torch.stack([noisy_stft.real, noisy_stft.imag], dim=-1)
        label = torch.stack([clean_stft.real, clean_stft.imag], dim=0)

        return x_input, label  # (F, T, C) & (C, F, T) where C = 2


class TestDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, window_length, hop_length, n_fft):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        self.noisy_list = os.listdir(self.noisy_dir)
        self.clean_list = os.listdir(self.clean_dir)

        self.window = torch.hann_window(window_length=window_length).pow(0.5)
        self.hop_length = hop_length
        self.n_fft = n_fft

    def __len__(self):
        return len(self.noisy_list)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_list[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_list[idx])

        noisy, fs = sf.read(noisy_path, dtype='float32')
        clean, _ = sf.read(clean_path, dtype='float32')

        # 记得用torch.tensor()把数据转成tensor
        noisy = torch.tensor(noisy)
        clean = torch.tensor(clean)

        noisy_stft = torch.stft(noisy, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window, return_complex=True)

        x_input = torch.stack([noisy_stft.real, noisy_stft.imag], dim=-1)

        return x_input, noisy, clean  # (F, T, C) where C = 2

















