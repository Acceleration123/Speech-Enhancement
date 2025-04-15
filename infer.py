import torch
import torch.nn as nn
import soundfile as sf
import os
import pandas as pd
import toml
from Model_Final import DL_TF_Grid
from pystoi import stoi
# from pesq import pesq
from collections import OrderedDict

configure = toml.load("configure.toml")


class infer(nn.Module):
    def __init__(self, model, device, state_dict):
        super(infer, self).__init__()
        self.model = model
        self.device = device
        self.state_dict = state_dict
        self.new_state_dict = OrderedDict()

        for k, v in self.state_dict.items():
            name = k[7:]  # remove `module.`
            self.new_state_dict[name] = v

        self.model.load_state_dict(self.new_state_dict)
        self.model.to(self.device).eval()
        self.window = torch.hann_window(configure['FFT']['win_length']).pow(0.5).to(self.device)

    def test_dataset_infer(self, noisy_dir_, save_dir_):
        noisy_list = os.listdir(noisy_dir_)

        file_length = 5

        for i in range(len(noisy_list)):
            noisy, fs = sf.read(os.path.join(noisy_dir_, noisy_list[i]), dtype='float32')
            noisy = torch.tensor(noisy).to(self.device)

            noisy_stft = torch.stft(noisy, n_fft=configure['FFT']['n_fft'], hop_length=configure['FFT']['hop_length'],
                                    window=self.window, return_complex=False).unsqueeze(0)  # (1, F, T, 2)

            pred_stft = self.model(noisy_stft).squeeze(0).permute(1, 2, 0)  # (F, T, 2)

            pred = torch.istft(pred_stft[:, :, 0] + 1j * pred_stft[:, :, 1], n_fft=configure['FFT']['n_fft'],
                               hop_length=configure['FFT']['hop_length'], window=self.window)
            pred = pred.cpu().detach().numpy()

            sf.write(os.path.join(save_dir_, f"{i:0{file_length}d}.wav"), pred, fs)

    def metrics(self, noisy_dir_, clean_dir_, pred_dir_):
        noisy_list_ = os.listdir(noisy_dir_)
        clean_list_ = os.listdir(clean_dir_)
        pred_list_ = os.listdir(pred_dir_)

        stoi_score = []
        # pesq_score = []

        for i in range(len(noisy_list_) - 1):
            noisy, fs = sf.read(os.path.join(noisy_dir_, noisy_list_[i]), dtype='float32')
            clean, _ = sf.read(os.path.join(clean_dir_, clean_list_[i]), dtype='float32')
            pred, _ = sf.read(os.path.join(pred_dir_, pred_list_[i]), dtype='float32')

            # STOI
            stoi_before = stoi(clean, noisy, fs, extended=False)
            stoi_after = stoi(clean, pred, fs, extended=False)
            print(stoi_before, stoi_after, stoi_after - stoi_before)

            new_stoi = {
                "Before": stoi_before,
                "After": stoi_after,
                "Improvement": stoi_after - stoi_before
            }
            stoi_score.append(new_stoi)

            """
            # PESQ
            pesq_before = pesq(fs, clean, noisy, 'wb')
            pesq_after = pesq(fs, clean, pred, 'wb')
            new_pesq = {
                "Before": pesq_before,
                "After": pesq_after,
                "Improvement": pesq_after - pesq_before
            }
            pesq_score.append(new_pesq)
            """

        stoi_score = pd.DataFrame(stoi_score)
        # pesq_score = pd.DataFrame(pesq_score)

        stoi_score.to_excel("stoi_score.xlsx")

        return stoi_score  #, pesq_score

    def single_speech_infer(self, noisy, fs):
        noisy = torch.tensor(noisy).to(self.device)

        noisy_stft = torch.stft(noisy, n_fft=configure['FFT']['n_fft'], hop_length=configure['FFT']['hop_length'],
                                window=self.window, return_complex=False).unsqueeze(0)  # (1, F, T, 2)
        pred_stft = self.model(noisy_stft).squeeze(0).permute(1, 2, 0)  # (F, T, 2)

        pred = torch.istft(pred_stft[:, :, 0] + 1j * pred_stft[:, :, 1], n_fft=configure['FFT']['n_fft'],
                           hop_length=configure['FFT']['hop_length'], window=self.window)
        pred = pred.cpu().detach().numpy()

        sf.write('pred-2.wav', pred, fs)


if __name__ == '__main__':
    noisy_dir = configure["test_dataset"]["noisy_dir"]
    clean_dir = configure["test_dataset"]["clean_dir"]
    pred_dir = configure["test_dataset"]["pred_dir"]

    model = DL_TF_Grid()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load("DL_TF_Grid.pth", map_location=device)

    infer = infer(model, device, state_dict)
    noisy, fs = sf.read('0008.wav', dtype='float32')
    infer.single_speech_infer(noisy, fs)







