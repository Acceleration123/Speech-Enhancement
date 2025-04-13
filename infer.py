import torch
import torch.nn as nn
import soundfile as sf
from Model_DL import DL_TF_Grid
import toml
import os

config = toml.load('configure.toml')
local_rank = int(os.environ["LOCAL_RANK"])

n_gpus = 2
torch.distributed.init_process_group("nccl", init_method="env://", world_size=n_gpus, rank=local_rank)
torch.cuda.set_device(local_rank)
device = torch.device(local_rank)

noisy, fs = sf.read('/data/ssd0/leyan.yang/DNS5/test_noisy/00199.wav', dtype='float32')
clean, _ = sf.read('/data/ssd0/leyan.yang/DNS5/test_clean/00199.wav', dtype='float32')

noisy = torch.tensor(noisy).to(device)
clean = torch.tensor(clean).to(device)

window = torch.hann_window(config['FFT']['win_length']).pow(0.5).to(device)
noisy_stft = torch.stft(noisy, n_fft=config['FFT']['n_fft'], hop_length=config['FFT']['hop_length'], 
                   window=window, return_complex=False)
clean_stft = torch.stft(clean, n_fft=config['FFT']['n_fft'], hop_length=config['FFT']['hop_length'], 
                   window=window, return_complex=False)  # (F, T, 2)

model = DL_TF_Grid().to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
model.load_state_dict(torch.load('DL_TF-Grid_model.pth', map_location=device))
model.eval()

noisy_stft = noisy_stft.to(device)

pred_stft = model(noisy_stft.unsqueeze(0))  # (1, 2, F, T)
pred_stft = pred_stft.squeeze(0)  # (2, F, T)
pred_stft_real, pred_stft_imag = pred_stft[0, :, :], pred_stft[1, :, :]

pred = torch.istft(pred_stft_real + 1j * pred_stft_imag, n_fft=config['FFT']['n_fft'],  
                   hop_length=config['FFT']['hop_length'], window=window)

pred = pred.cpu().detach().numpy()
clean = clean.cpu().detach().numpy()
noisy = noisy.cpu().detach().numpy()

sf.write('pred.wav', pred, fs)
sf.write('noisy.wav', noisy, fs)
sf.write('clean.wav', clean, fs)


