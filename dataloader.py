from torch.utils import data
import librosa
import random
import numpy as np
import pandas as pd
import soundfile as sf


TRAIN_CLEAN_PATH = "D:\\Users\\14979\\Desktop\\DNS5_16k\\train_clean"
TRAIN_NOISY_PATH = "D:\\Users\\14979\\Desktop\\DNS5_16k\\train_noisy"
TRAIN_NOISE_CSV_PATH = "D:\\Users\\Database\\train_noise_data_new.csv"
VALID_CLEAN_PATH = "D:\\Users\\14979\\Desktop\\DNS5_16k\\dev_clean"
VALID_NOISY_PATH = "D:\\Users\\14979\\Desktop\\DNS5_16k\\dev_noisy"


def mk_mixture(clean, noise, snr):
    scale_factor = np.sqrt(np.var(clean) * (10 ** (-snr / 10)) / (np.var(noise + 1e-8)))
    noisy = clean + scale_factor * noise
    return noisy


class Train_Dataset(data.Dataset):
    def __init__(
            self,
            fs=16000,
            length_in_seconds=8,
            num_tot=60000,
            num_per_epoch=10000,
            random_sample=True,
            random_start=True,
            middle_blks=4,
            snr_delta=5,
            pl=False
    ):
        super(Train_Dataset, self).__init__()
        self.train_clean_database = sorted(librosa.util.find_files(TRAIN_CLEAN_PATH, ext='wav'))[:num_tot]
        self.train_noisy_database = sorted(librosa.util.find_files(TRAIN_NOISY_PATH, ext='wav'))[:num_tot]
        self.train_noise_database = sorted(pd.read_csv(TRAIN_NOISE_CSV_PATH)['file_dir'].tolist())[:num_tot]

        self.L = int(fs * length_in_seconds)
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_tot = num_tot
        self.num_per_epoch = num_per_epoch
        self.random_sample = random_sample
        self.random_start = random_start
        self.middle_blks = middle_blks
        self.snr_delta = snr_delta
        self.pl = pl

    def __len__(self):
        return self.num_per_epoch

    def __getitem__(self, idx):
        # 训练的时候随机采样
        if self.pl:
            clean_list = random.sample(self.train_clean_database, self.num_per_epoch)
            noise_list = random.sample(self.train_noise_database, self.num_per_epoch)
            # 裁剪
            if self.random_start:
                begin_s = int(self.fs * random.uniform(0, 10 - self.length_in_seconds))
                label, _ = sf.read(clean_list[idx], dtype='float32', start=begin_s, stop=begin_s + self.L)
                noise, _ = sf.read(noise_list[idx], dtype='float32', start=begin_s, stop=begin_s + self.L)
            else:
                label, _ = sf.read(clean_list[idx], dtype='float32', start=0, stop=self.L)
                noise, _ = sf.read(noise_list[idx], dtype='float32', start=0, stop=self.L)

            noisy = mk_mixture(label, noise, random.uniform(-5, 15))

            label_final = None
            for k in range(self.middle_blks):
                if label_final is None:
                    label_final = mk_mixture(label, noise, k * self.snr_delta)[:, None]
                else:
                    label_final = np.concatenate([label_final, mk_mixture(label, noise, k * self.snr_delta)[:, None]], axis=1)

            label_final = np.concatenate([label_final, label[:, None]], axis=1)

        else:
            clean_list = random.sample(self.train_clean_database, self.num_per_epoch)
            noisy_list = random.sample(self.train_noisy_database, self.num_per_epoch)

            # 裁剪
            if self.random_start:
                begin_s = int(self.fs * random.uniform(0, 10 - self.length_in_seconds))
                label_final, _ = sf.read(clean_list[idx], dtype='float32', start=begin_s, stop=begin_s + self.L)
                noisy, _ = sf.read(noisy_list[idx], dtype='float32', start=begin_s, stop=begin_s + self.L)
            else:
                label_final, _ = sf.read(clean_list[idx], dtype='float32', start=0, stop=self.L)
                noisy, _ = sf.read(noisy_list[idx], dtype='float32', start=0, stop=self.L)

        return noisy, label_final  # (T,) & (T,) in direct learning mode, (T, 5) in progressive learning mode


class Valid_Dataset(data.Dataset):
    def __init__(self):
        self.valid_clean_database = sorted(librosa.util.find_files(VALID_CLEAN_PATH, ext='wav'))
        self.valid_noisy_database = sorted(librosa.util.find_files(VALID_NOISY_PATH, ext='wav'))

    def __len__(self):
        return len(self.valid_clean_database)

    def __getitem__(self, idx):
        label, _ = sf.read(self.valid_clean_database[idx], dtype='float32')
        noisy, _ = sf.read(self.valid_noisy_database[idx], dtype='float32')

        return noisy, label  # (T,) & (T,)


if __name__ == '__main__':
    # test train_dataset in direct learning mode
    train_dataset_dl = Train_Dataset(num_per_epoch=10, pl=False)
    train_loader_dl = data.DataLoader(train_dataset_dl, batch_size=1, shuffle=True, num_workers=0)
    for i, (noisy, label) in enumerate(train_loader_dl):
        print(i, noisy.shape, label.shape)

    # test train_dataset in progressive learning mode
    train_dataset_pl = Train_Dataset(num_per_epoch=10, pl=True)
    train_loader_pl = data.DataLoader(train_dataset_pl, batch_size=1, shuffle=True, num_workers=0)
    for i, (noisy, label) in enumerate(train_loader_pl):
        print(i, noisy.shape, label.shape)
        num = label.shape[-1]
        if i == 0:
            label = label.squeeze(0)
            for j in range(num):
                sf.write(f"label_{j}.wav", label[:, j].cpu().detach().numpy(), 16000)

    # test valid_dataset
    valid_dataset = Valid_Dataset()
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (noisy, label) in enumerate(valid_loader):
        print(i, noisy.shape, label.shape)

