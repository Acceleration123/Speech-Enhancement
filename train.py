import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
import toml
import Model_DL
from dataset import TrainDataset_DL
import loss_tf
import loss
import os


def run(epochs, local_rank, train_dataset, validation_dataset, model, criterion, optimizer):
    device = torch.device(local_rank)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  # 转移到多个GPU上去训练
    criterion.to(device)

    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    validation_sampler = torch.utils.data.DistributedSampler(validation_dataset)

    train_loader = data.DataLoader(train_dataset, sampler=train_sampler, **config['train_dataloader'])
    validation_loader = data.DataLoader(validation_dataset, sampler=validation_sampler, **config['validation_dataloader'])
    # warm_up = torch.tensor(30000, dtype=torch.float32)
    # train_steps = torch.tensor(1, dtype=torch.float32)
    # lr = optimizer.param_groups[0]['lr']

    num_steps = 0

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_total_loss = 0
        validation_total_loss = 0
        cnt_1 = -1
        cnt_2 = -1
        eps = 1e-6
        model.train()  # 训练模式
        for noisy_stft, clean_stft in train_loader:
            """
            optimizer.param_groups[0]['lr'] = max(lr * min(1 / train_steps, train_steps / warm_up), torch.tensor(0.001, dtype=torch.float32))
            """

            if num_steps > 200:
             optimizer.param_groups[0]['lr'] = 0.0001

            # 前向传播
            noisy_stft, clean_stft = noisy_stft.to(device), clean_stft.to(device)
            outputs = model(noisy_stft)
            train_loss = criterion(outputs, clean_stft)

            # 反向传播
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_total_loss += train_loss.item()
            num_steps += 1

        torch.save(model.state_dict(), "DL_TF-Grid.pth")
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_total_loss / len(train_loader):.4f}")

        model.eval()  # 验证模式
        with torch.no_grad():
            for noisy_stft, clean_stft in validation_loader:
                # 前向传播 验证无需要反向传播！
                noisy_stft, clean_stft = noisy_stft.to(device), clean_stft.to(device)
                outputs = model(noisy_stft)
                validation_loss = criterion(outputs, clean_stft)
                validation_total_loss += validation_loss.item()

            val_loss = validation_total_loss / len(validation_loader)

            if epoch == 0:
                check_loss = val_loss

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {val_loss:.4f}")

        if np.abs(check_loss - val_loss) < eps:
            cnt_1 += 1
            cnt_2 += 1
            check_loss = val_loss

        if cnt_1 == 5:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
            cnt_1 = 0

        if cnt_2 == 10:
            break


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])

    n_gpus = 2
    torch.distributed.init_process_group("nccl", init_method="env://", world_size=n_gpus, rank=local_rank)
    torch.cuda.set_device(local_rank)

    config = toml.load('configure.toml')

    # 加载数据集
    train_dataset = TrainDataset_DL(**config['train_dataset'], window_length=config['FFT']['win_length'],
                                    hop_length=config['FFT']['hop_length'], n_fft=config['FFT']['n_fft'])

    validation_dataset = TrainDataset_DL(**config['validation_dataset'], window_length=config['FFT']['win_length'],
                                          hop_length=config['FFT']['hop_length'], n_fft=config['FFT']['n_fft'])

    # 加载模型
    model = Model_DL.DL_TF_Grid()

    # 加载损失函数和优化器
    criterion = loss.loss_totally()
    optimizer = optim.Adam(model.parameters(), **config['optimizer'])

    # 训练模型
    epochs = 30

    run(epochs, local_rank, train_dataset, validation_dataset, model, criterion, optimizer)

