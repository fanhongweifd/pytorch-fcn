#!/usr/bin/env python
import argparse
import datetime
import os
import os.path as osp
import torch.optim.lr_scheduler as lr_scheduler
import torch
import yaml
import torchfcn


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    parser.add_argument(
        '--max-epoch', type=int, default=200, help='max epoch'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-14, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument('--train_data_file', type=str, help='data_file_excel')
    parser.add_argument('--train_label_file', type=str, help='label_file_excel')
    parser.add_argument('--valid_data_file', type=str, help='data_file_excel')
    parser.add_argument('--valid_label_file', type=str, help='label_file_excel')
    parser.add_argument('--file_type', type=str, default='xlsx' ,help='xlsx or pickle')
    parser.add_argument('--feature_dim', default=87, help='label_file_excel')
    
    args = parser.parse_args()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.TransportData(args.train_data_file, args.train_label_file,file_type=args.file_type, feature_dim=args.feature_dim),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.TransportData(args.valid_data_file, args.valid_label_file,file_type=args.file_type, feature_dim=args.feature_dim),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN8sPM25_1conv()
    
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    max_epoch = args.max_epoch
    args.max_iteration = max_epoch * len(train_loader)
    print('max_epoch=%s'% max_epoch )
    print('max_iter=%s'%args.max_iteration)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=max_epoch , eta_min = 0.000000001)
    # scheduler = lr_scheduler.ExponentialLR(optim, 0.9)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
    print('load train data  = %s' % len(train_loader))
    print('load val data  = %s' % len(val_loader))
    
    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=10,
        use_grad_clip=True,
        clip=50
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
