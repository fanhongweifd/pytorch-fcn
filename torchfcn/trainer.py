import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

def sum_gredient(model):
    """
    功能函数,用来观察梯度变化
    :param model:
    :param lr:
    """
    sum_gred = 0
    num_gred = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            num_gred += torch.numel(m.weight.grad)
            sum_gred += torch.abs(m.weight.grad).sum()
    avg_grad = sum_gred/num_gred
    return avg_grad

def mseloss_2d(input, target, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    input_reshape = input.transpose(1, 2).transpose(2, 3).contiguous()
    target = target.view(n, h, w, 1).repeat(1, 1, 1, c)
    mask = target > 0
    loss = F.mse_loss(input_reshape.float(), target.float(), size_average=None, reduce=None, reduction='none')
    loss = loss.float() * mask.float()
    loss = torch.sum(loss.float())
    if size_average:
        if mask.data.sum() > 0:
            loss /= mask.data.sum()
    return loss

def smape(A, F):
        return 100 / len(A) * np.sum( np.abs(F - A) / (np.abs(A) + np.abs(F)))

def smape_loss(input, target):
    n, c, h, w = input.size()
    # print('input = %s',input)
    # print('target = %s', target)
    try:
        input_reshape = input.transpose(1, 2).transpose(2, 3).contiguous()
        target = target.view(n, h, w, 1).repeat(1, 1, 1, c)
        mask = target > 0
        input_array = input_reshape[mask].detach().numpy()
        target_array = target[mask].numpy()
        if mask.data.sum() <= 0:
            smape_value = np.nan
            # print('predict_array mask = %s' % input_array)
            # print('target_array mask = %s' % target_array)
            return smape_value
        smape_value = smape(input_array, target_array)
    except Exception as e:
        print(e)
        raise ValueError
    # print('predict_array mask = %s' % input_array)
    # print('target_array mask = %s' % target_array)
    return smape_value


class Trainer(object):

    def __init__(self, cuda, model, optimizer, scheduler,
                 train_loader, val_loader, out, max_iter,
                 size_average=True, interval_validate=None, use_scheduler=True, use_grad_clip=True, clip=1):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now()
        self.size_average = size_average
        self.use_scheduler = use_scheduler
        self.use_grad_clip = use_grad_clip
        self.clip = clip
        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'lr',
            'train/loss',
            'train/smape',
            'valid/loss',
            'valid/smape',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_smape = 0

    def validate(self):
        training = self.model.training
        self.model.eval()
        val_loss = 0
        val_smape_loss = 0
        for batch_idx, (data, target) in enumerate(self.val_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                score = self.model(data)

            loss = mseloss_2d(score, target, size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            smape_loss_data = smape_loss(score, target)
            # print('this batch smape_loss =%s, all val smape_loss=%s'%(smape_loss_data,val_smape_loss))
            # 这里没有把loss为nan的情况在总数中减去，后面要加入
            if np.isnan(smape_loss_data):
                continue
            val_smape_loss += smape_loss_data
            val_loss += loss_data

        val_loss /= len(self.val_loader)
        print('smape loss = %s' % val_smape_loss)
        val_smape_loss /= len(self.val_loader)
        print('avg smape loss = %s'%val_smape_loss)
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now() -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [self.optim.param_groups[0]['lr']] + [''] * 2 + \
                  [val_loss] + [val_smape_loss] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        print('val epoch = %s, iteration=%s, lr=%s, val_loader_num=%s, val_mce_loss=%s, val_smape=%s' % (
        self.epoch, self.iteration, self.optim.param_groups[0]['lr'], len(self.val_loader), val_loss, val_smape_loss))
        is_best = val_smape_loss < self.best_smape
        if is_best:
            self.best_smape = val_smape_loss
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_smape': self.best_smape,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        # for batch_idx, (data, target) in tqdm.tqdm(
        #         enumerate(self.train_loader), total=len(self.train_loader),
        #         desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # target 不能全为空
            # mask = target > 0
            # if mask.data.sum()<=0:
            #     continue
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration > 0 and self.iteration % (len(self.train_loader)*self.interval_validate) == 0:
                print('Begin valid:')
                self.validate()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)
            smape_loss_data = smape_loss(score, target)
            loss = mseloss_2d(score, target, size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            avg_grad = sum_gredient(self.model)
            self.optim.step()
            

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now() -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [self.optim.param_groups[0]['lr']] + [loss_data] + \
                    [smape_loss_data] + [''] * 2 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')
                print('train epoch = %s, iteration=%s, lr=%s, avg_grad=%s, mce_loss=%s, smape=%s'%(
                self.epoch, self.iteration, self.optim.param_groups[0]['lr'], avg_grad.data.item(), loss_data, smape_loss_data))
            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        # for epoch in tqdm.trange(self.epoch, max_epoch,
        #                          desc='Train', ncols=80):
        for epoch in range(self.epoch, max_epoch):
            if self.use_scheduler:
                self.scheduler.step(epoch)
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

if __name__ == '__main__':
    input = torch.randn(2, 1, 5, 5, requires_grad=True)
    target = torch.randn(2, 5, 5)
    loss = mseloss_2d(input, target)
    print(loss)
    loss.backward()
    print(input.grad)
    print(target)
    smape = smape_loss(input, target)
    print(smape)