import torch
import numpy as np
import soundfile as sf
import os
import pandas as pd
import concurrent.futures
import argparse
import glob
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm

from model.Model_DL import DL_TF_Grid as Model
from DNSMOS.dnsmos_local import ComputeScore


class infer:
    def __init__(self, model, device):
        super(infer, self).__init__()
        self.model = model
        self.device = device

    def test_dataset_infer(self, args):
        noisy_list = os.listdir(args.noisy_dir)

        file_length = 5

        for i in tqdm(range(len(noisy_list))):
            noisy, fs = sf.read(os.path.join(args.noisy_dir, noisy_list[i]), dtype='float32')
            noisy = torch.tensor(noisy).to(self.device).unsqueeze(0)  # (1, T)

            pred = self.model(noisy).squeeze(0)  # (T,)
            pred = pred.cpu().detach().numpy()

            sf.write(os.path.join(args.pred_dir, f"{i:0{file_length}d}.wav"), pred, fs)

    def get_sisnr(self, ref, inf):
        inf = inf - inf.mean()
        ref = ref - ref.mean()

        a = np.sum(inf * ref) / np.sum(ref ** 2 + 1e-8)
        e_tar = a * ref
        e_res = inf - e_tar

        sisnr = 10 * np.log10((np.sum(e_tar ** 2) + 1e-8) / (np.sum(e_res ** 2) + 1e-8))

        return sisnr

    def intrusive_metrics(self, args):
        noisy_list_ = os.listdir(args.noisy_dir)
        clean_list_ = os.listdir(args.clean_dir)
        pred_list_ = os.listdir(args.pred_dir)

        stoi_score = []
        pesq_score = []
        sisnr_score = []

        for i in tqdm(range(len(noisy_list_))):
            noisy, fs = sf.read(os.path.join(args.noisy_dir, noisy_list_[i]), dtype='float32')
            clean, _ = sf.read(os.path.join(args.clean_dir, clean_list_[i]), dtype='float32')
            pred, _ = sf.read(os.path.join(args.pred_dir, pred_list_[i]), dtype='float32')

            # STOI
            stoi_before = stoi(clean, noisy, fs, extended=False)
            stoi_after = stoi(clean, pred, fs, extended=False)

            stoi_score.append({
                'Before': stoi_before,
                'After': stoi_after,
                'Improvement': stoi_after - stoi_before
            })

            # PESQ
            pesq_before = pesq(fs, clean, noisy, 'wb')
            pesq_after = pesq(fs, clean, pred, 'wb')

            pesq_score.append({
                'Before': pesq_before,
                'After': pesq_after,
                'Improvement': pesq_after - pesq_before
            })

            #SISNR
            sisnr_before = self.get_sisnr(clean, noisy)
            sisnr_after = self.get_sisnr(clean, pred)

            sisnr_score.append({
                'Before': sisnr_before,
                'After': sisnr_after,
                'Improvement': sisnr_after - sisnr_before
            })

        stoi_score = pd.DataFrame(stoi_score)
        pesq_score = pd.DataFrame(pesq_score)
        sisnr_score = pd.DataFrame(sisnr_score)

        stoi_score.to_excel("stoi_score.xlsx")
        pesq_score.to_excel("pesq_score.xlsx")
        sisnr_score.to_excel("sisnr_score.xlsx")

    def dnsmos_metrics(self, args, fs=16000):
        models = glob.glob(os.path.join(args.testset_dir, "*"))
        p808_model_path = os.path.join('DNSMOS/DNSMOS', 'model_v8.onnx')

        if args.personalized_MOS:
            primary_model_path = os.path.join('DNSMOS/pDNSMOS', 'sig_bak_ovr.onnx')
        else:
            primary_model_path = os.path.join('DNSMOS/DNSMOS', 'sig_bak_ovr.onnx')

        compute_score = ComputeScore(primary_model_path, p808_model_path)

        rows = []
        clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
        is_personalized_eval = args.personalized_MOS
        desired_fs = fs
        
        for m in tqdm(models):
            max_recursion_depth = 10
            audio_path = os.path.join(args.testset_dir, m)
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        
            while len(audio_clips_list) == 0 and max_recursion_depth > 0:
                audio_path = os.path.join(audio_path, "**")
                audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
                max_recursion_depth -= 1
            clips.extend(audio_clips_list)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip for clip in
                             clips}
            for future in tqdm(concurrent.futures.as_completed(future_to_url)):
                clip = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (clip, exc))
                else:
                    rows.append(data)

        df = pd.DataFrame(rows)
        if args.csv_path:
            csv_path = args.csv_path
            df.to_csv(csv_path)
        else:
            print(df.describe())

    def single_speech_infer(self, noisy, fs):
        noisy = torch.tensor(noisy).to(self.device).unsqueeze(0)  # (1, T)
        pred = self.model(noisy).squeeze(0)  # (T,)
        pred = pred.cpu().detach().numpy()

        sf.write('pred.wav', pred, fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--noisy_dir', type=str,
                        help = 'path of noisy speech directory',
                        default="C:\\Users\\86151\\Desktop\\DNS5_16k\\DNS5_16k\\test_noisy"
                        )

    parser.add_argument('-c', '--clean_dir', type=str,
                        help = 'path of clean speech directory',
                        default="C:\\Users\\86151\\Desktop\\DNS5_16k\\DNS5_16k\\test_clean"
                        )

    parser.add_argument('-pr', '--pred_dir', type=str,
                        help = 'path of predicted speech directory',
                        default="C:\\Users\\86151\\Desktop\\NN-Zoo\\dual_unet\\test_pred"
                        )

    parser.add_argument('-t', "--testset_dir",
                        help='Path to the dir containing audio clips in .wav to be evaluated',
                        default="C:\\Users\\86151\\Desktop\\NN-Zoo\\dual_unet\\test_pred")

    parser.add_argument('-o', "--csv_path",
                        help='Dir to the csv that saves the results',
                        default="test_results.csv"
                        )

    parser.add_argument('-p', "--personalized_MOS",
                        action='store_true',
                        help='Flag to indicate if personalized MOS score is needed or regular'
                        )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    state_dict = torch.load("best_model.tar", map_location=device)
    model.load_state_dict(state_dict['model'])
    model.eval()

    infer = infer(model, device)
    infer.test_dataset_infer(args)
    infer.dnsmos_metrics(args)
    infer.intrusive_metrics(args)

