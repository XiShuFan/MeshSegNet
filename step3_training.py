import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_dataset import *
from meshsegnet_ln import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Union
import copy
import datetime

use_visdom = True  # if you don't use visdom, please set to False

def main():
    # 多进程多卡训练
    ngpus_per_node = torch.cuda.device_count()
    print(f'total gpus {ngpus_per_node}')
    if ngpus_per_node >= 2:
        world_size = ngpus_per_node
        mp.spawn(main_worker, args=(world_size,), nprocs=ngpus_per_node, join=True)
    else:
        main_worker(0, 1)


def setup(rank, world_size):
    if world_size >= 2:
        # 分布式需要调用下面的语句
        # ubuntu下 nccl，windows下 gloo
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12345', world_size=world_size, rank=rank)
    else:
        return


def reduce_sum(metric: torch.tensor, world_size: int):
    """
    将多卡计算的指标加起来
    """
    if world_size >= 2:
        metric_tensor = torch.tensor(metric).cuda()
        # all_reduce可以自动同步
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        return metric_tensor.item()
    else:
        return metric


def main_worker(rank, world_size):
    print(f'process using gpu {rank}')

    if use_visdom:
        plotter = utils.VisdomLinePlotter(env_name='MeshSegNet')

    setup(rank, world_size)

    # 指定使用的gpu，后面就可以直接使用.cuda()
    torch.cuda.set_device(rank)

    train_list = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan/train_list_1.csv'
    val_list = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan/val_list_1.csv'

    # train_list = 'D:\\users\\xsf\\Dataset\\OralScan\\train_list_1.csv'
    # val_list = 'D:\\users\\xsf\\Dataset\\OralScan\\val_list_1.csv'

    model_path = 'models/13/'
    model_name = 'MeshSegNet_15_classes'  # need to define
    checkpoint_name = 'latest_checkpoint.tar'

    num_classes = 17
    num_channels = 18  # number of features
    num_epochs = 600
    num_workers = 8
    train_batch_size = 1
    val_batch_size = 1
    num_batches_to_print = 30
    # 加载预训练模型
    load_pretrain = False
    patch_size = 9000

    # mkdir 'models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # set dataset
    training_dataset = Mesh_Dataset(data_list_path=train_list,
                                    num_classes=num_classes,
                                    patch_size=patch_size,
                                    device=f'cuda:{rank}')

    val_dataset = Mesh_Dataset(data_list_path=val_list,
                               num_classes=num_classes,
                               patch_size=patch_size,
                               device=f'cuda:{rank}')

    if world_size >= 2:
        training_sampler = DistributedSampler(training_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        training_sampler = None
        val_sampler = None

    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              sampler=training_sampler,
                              num_workers=num_workers,
                              drop_last=True,
                              shuffle=training_sampler is None)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            drop_last=True,
                            shuffle=val_sampler is None)

    # 获得模型
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5, cells=patch_size, feature_base=32)
    # 优化器
    opt = optim.Adam(model.parameters(), amsgrad=True, lr=1e-3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count ', total_params)

    if load_pretrain:
        # 加载预训练模型，注意要加载到CPU上，否则可能会导致加载的权重全部集中在一张卡上
        checkpoint = torch.load('./models/MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip', map_location='cpu')
        # 加载需要的部分权重
        model.load_state_dict(checkpoint['model_state_dict'])


    if world_size >= 2:
        # 转换成进程间同步的SyncBatchNorm层，缓解batch size较小的问题
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        model = model.cuda()


    # 记录每个epoch的loss和指标
    start_times, end_times, losses, mdsc, msen, mppv = [], [], [], [], [], []
    val_start_times, val_end_times, val_losses, val_mdsc, val_msen, val_mppv = [], [], [], [], [], []

    best_val_dsc = 0.0

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Training model...')

    # 类别权重
    class_weights = torch.ones(num_classes).cuda().float()
    for epoch in range(num_epochs):
        if world_size >= 2:
            # 让数据shuffle在每一个epoch都有效
            training_sampler.set_epoch(epoch)

        # training
        model.train()
        running_loss =0.0
        running_mdsc =0.0
        running_msen =0.0
        running_mppv =0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        # 训练epoch开始时间
        start_times.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        for i_batch, batched_sample in enumerate(train_loader):
            # send mini-batch to device
            inputs = batched_sample['cells'].cuda().float()
            labels = batched_sample['labels'].cuda().long()
            A_S = batched_sample['A_S'].cuda().float()
            A_L = batched_sample['A_L'].cuda().float()

            # 标签转换成独热码
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, A_S, A_L)

            # 计算损失的时候还考虑了类别权重
            # loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            loss = CELoss(outputs, one_hot_labels)

            # 计算加权之后的评价指标
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()

            # 没有使用学习率调度
            opt.step()

            # print statistics
            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()



            # 只在gpu0上处理输出
            if i_batch % num_batches_to_print == num_batches_to_print - 1:
                if use_visdom:
                    plotter.plot(f'loss {rank}', 'train', 'Loss', epoch + (i_batch + 1) / len(train_loader), running_loss / num_batches_to_print)
                    plotter.plot(f'DSC {rank}', 'train', 'DSC', epoch + (i_batch + 1) / len(train_loader), running_mdsc / num_batches_to_print)
                    plotter.plot(f'SEN {rank}', 'train', 'SEN', epoch + (i_batch + 1) / len(train_loader), running_msen / num_batches_to_print)
                    plotter.plot(f'PPV {rank}', 'train', 'PPV', epoch + (i_batch + 1) / len(train_loader), running_mppv / num_batches_to_print)

                # 统计多卡的指标
                loss_mean = reduce_sum(running_loss, world_size) / (num_batches_to_print * world_size)
                mdsc_mean = reduce_sum(running_mdsc, world_size) / (num_batches_to_print * world_size)
                msen_mean = reduce_sum(running_msen, world_size) / (num_batches_to_print * world_size)
                mppv_mean = reduce_sum(running_mppv, world_size) / (num_batches_to_print * world_size)
                if rank == 0:
                    msg = '[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}'.format(
                        epoch + 1,
                        num_epochs,
                        i_batch + 1,
                        len(train_loader),
                        loss_mean,
                        mdsc_mean,
                        msen_mean,
                        mppv_mean
                    )
                    print(msg)
                    if use_visdom:
                        plotter.plot('loss', 'train', 'Loss', epoch + (i_batch + 1) / len(train_loader), loss_mean)
                        plotter.plot('DSC', 'train', 'DSC', epoch + (i_batch + 1) / len(train_loader), mdsc_mean)
                        plotter.plot('SEN', 'train', 'SEN', epoch + (i_batch + 1) / len(train_loader), msen_mean)
                        plotter.plot('PPV', 'train', 'PPV', epoch + (i_batch + 1) / len(train_loader), mppv_mean)
                running_loss = 0.0
                running_mdsc = 0.0
                running_msen = 0.0
                running_mppv = 0.0

        # record losses and metrics
        # 训练epoch结束时间
        end_times.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        losses.append(reduce_sum(loss_epoch, world_size) / len(train_loader) / world_size)
        mdsc.append(reduce_sum(mdsc_epoch, world_size) / len(train_loader) / world_size)
        msen.append(reduce_sum(msen_epoch, world_size) / len(train_loader) / world_size)
        mppv.append(reduce_sum(mppv_epoch, world_size) / len(train_loader) / world_size)

        # 每个epoch就验证一次
        # validation
        model.eval()

        # 验证epoch开始时间
        val_start_times.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        with torch.no_grad():
            running_val_loss = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):
                # send mini-batch to device
                inputs = batched_val_sample['cells'].cuda().float()
                labels = batched_val_sample['labels'].cuda().long()
                A_S = batched_val_sample['A_S'].cuda().float()
                A_L = batched_val_sample['A_L'].cuda().float()
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                outputs = model(inputs, A_S, A_L)
                # loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                loss = CELoss(outputs, one_hot_labels)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_val_loss += loss.item()
                running_val_mdsc += dsc.item()
                running_val_msen += sen.item()
                running_val_mppv += ppv.item()
                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()



                if i_batch % num_batches_to_print == num_batches_to_print - 1:
                    # 统计多张卡的指标
                    val_loss_mean = reduce_sum(running_val_loss, world_size) / (num_batches_to_print * world_size)
                    val_mdsc_mean = reduce_sum(running_val_mdsc, world_size) / (num_batches_to_print * world_size)
                    val_msen_mean = reduce_sum(running_val_msen, world_size) / (num_batches_to_print * world_size)
                    val_mppv_mean = reduce_sum(running_val_mppv, world_size) / (num_batches_to_print * world_size)
                    if rank == 0:
                        msg = '[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}'.format(
                            epoch + 1,
                            num_epochs,
                            i_batch + 1,
                            len(val_loader),
                            val_loss_mean,
                            val_mdsc_mean,
                            val_msen_mean,
                            val_mppv_mean
                        )
                        print(msg)

                    running_val_loss = 0.0
                    running_val_mdsc = 0.0
                    running_val_msen = 0.0
                    running_val_mppv = 0.0

            # record losses and metrics
            # 验证epoch结束时间
            val_end_times.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            val_losses.append(reduce_sum(val_loss_epoch, world_size) / len(val_loader) / world_size)
            val_mdsc.append(reduce_sum(val_mdsc_epoch, world_size) / len(val_loader) / world_size)
            val_msen.append(reduce_sum(val_msen_epoch, world_size) / len(val_loader) / world_size)
            val_mppv.append(reduce_sum(val_mppv_epoch, world_size) / len(val_loader) / world_size)

        # 输出当前epoch的结果
        if use_visdom:
            plotter.plot(f'loss {rank}', 'train', 'Loss', epoch + 1, losses[-1])
            plotter.plot(f'DSC {rank}', 'train', 'DSC', epoch + 1, mdsc[-1])
            plotter.plot(f'SEN {rank}', 'train', 'SEN', epoch + 1, msen[-1])
            plotter.plot(f'PPV {rank}', 'train', 'PPV', epoch + 1, mppv[-1])
            plotter.plot(f'loss {rank}', 'val', 'Loss', epoch + 1, val_losses[-1])
            plotter.plot(f'DSC {rank}', 'val', 'DSC', epoch + 1, val_mdsc[-1])
            plotter.plot(f'SEN {rank}', 'val', 'SEN', epoch + 1, val_msen[-1])
            plotter.plot(f'PPV {rank}', 'val', 'PPV', epoch + 1, val_mppv[-1])


        if rank == 0:
            msg = '*****\nEpoch: {}/{}, start: {}, end: {}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         ' \
                  'start: {}, end: {}, val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(
                epoch + 1,
                num_epochs,
                start_times[-1],
                end_times[-1],
                losses[-1],
                mdsc[-1],
                msen[-1],
                mppv[-1],
                val_start_times[-1],
                val_end_times[-1],
                val_losses[-1],
                val_mdsc[-1],
                val_msen[-1],
                val_mppv[-1]
            )
            print(msg)
            if use_visdom:
                plotter.plot('loss', 'train', 'Loss', epoch + 1, losses[-1])
                plotter.plot('DSC', 'train', 'DSC', epoch + 1, mdsc[-1])
                plotter.plot('SEN', 'train', 'SEN', epoch + 1, msen[-1])
                plotter.plot('PPV', 'train', 'PPV', epoch + 1, mppv[-1])
                plotter.plot('loss', 'val', 'Loss', epoch + 1, val_losses[-1])
                plotter.plot('DSC', 'val', 'DSC', epoch + 1, val_mdsc[-1])
                plotter.plot('SEN', 'val', 'SEN', epoch + 1, val_msen[-1])
                plotter.plot('PPV', 'val', 'PPV', epoch + 1, val_mppv[-1])

            # 训练完一个epoch，保存checkpoint
            # save the checkpoint
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict() if world_size >= 2 else model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'val_losses': val_losses,
                        'val_mdsc': val_mdsc,
                        'val_msen': val_msen,
                        'val_mppv': val_mppv},
                        os.path.join(model_path, checkpoint_name))

            # save the best model
            if best_val_dsc < val_mdsc[-1]:
                best_val_dsc = val_mdsc[-1]
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.module.state_dict() if world_size >= 2 else model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'losses': losses,
                            'mdsc': mdsc,
                            'msen': msen,
                            'mppv': mppv,
                            'val_losses': val_losses,
                            'val_mdsc': val_mdsc,
                            'val_msen': val_msen,
                            'val_mppv': val_mppv},
                            os.path.join(model_path, '{}_best.tar'.format(model_name)))

            # save all losses and metrics data
            pd_dict = {'start time': start_times, 'end time': end_times, 'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv,
                       'val start time': val_start_times, 'val end time': val_end_times, 'val_loss': val_losses, 'val_DSC': val_mdsc,
                       'val_SEN': val_msen, 'val_PPV': val_mppv}
            stat = pd.DataFrame(pd_dict)
            stat.to_csv(os.path.join(model_path, 'losses_metrics_vs_epoch.csv'))


if __name__ == '__main__':
    main()
