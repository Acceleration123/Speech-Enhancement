import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils import data

from dataloader import Train_Dataset as Train_Dataset
from dataloader import Valid_Dataset as Valid_Dataset
from model.Model_PL_end2end import PL_TF_Grid as Model
from loss_func import loss_hybrid as Loss
from scheduler import LinearWarmupCosineAnnealingLR as Scheduler
from trainer import Trainer as Trainer


seed = 43
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def run(rank, args):
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)

    collate_fn = Train_Dataset.collate_fn if hasattr(Train_Dataset, "collate_fn") else None
    shuffle = False if args.world_size > 1 else True

    config = OmegaConf.load(args.config)
    train_dataset = Train_Dataset(**config['train_dataset'])
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = data.DataLoader(
        train_dataset,
        **config['train_dataloader'],
        sampler=train_sampler,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

    valid_dataset = Valid_Dataset()
    valid_sampler = data.distributed.DistributedSampler(valid_dataset) if args.world_size > 1 else None
    valid_dataloader = data.DataLoader(
        valid_dataset,
        **config['valid_dataloader'],
        sampler=valid_sampler,
        collate_fn=collate_fn,
        shuffle=False
    )

    model = Model().to(args.device)
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), **config['optimizer'])
    scheduler = Scheduler(optimizer, **config['scheduler'])

    loss_func = Loss(pl=True).to(args.device)

    trainer = Trainer(config=config, model=model, optimizer=optimizer, scheduler=scheduler, loss_func=loss_func,
                      train_dataloader=train_dataloader, validation_dataloader=valid_dataloader,
                      train_sampler=train_sampler, args=args)
    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()  # 关闭进程组


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-C',
        '--config',
        type=str,  # 默认值也是str类型
        default='./configs/train_configure.yaml',
        help='training configure file path'
    )

    parser.add_argument(
        '-D',
        '--device',
        type=str,
        default='0',
        help='The index of the available devices, e.g. 0,1,2,3'
    )

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.world_size = len(args.device.split(','))

    if args.world_size > 1:
        torch.multiprocessing.spawn(run, args=(args,), nprocs=args.world_size, join=True)
    else:
        run(0, args)
