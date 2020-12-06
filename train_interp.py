import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from data.interpolation_data import KittiInterpolationDataset
from models.models import PointINet
from models.utils import ClippedStepLR, chamfer_loss

from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PointINet')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    parser.add_argument('--step_size_lr', type=int, default=10)
    parser.add_argument('--gamma_lr', type=float, default=0.7)
    parser.add_argument('--init_bn_momentum', type=float, default=0.5)
    parser.add_argument('--min_bn_momentum', type=float, default=0.01)
    parser.add_argument('--step_size_bn_momentum', type=int, default=10)
    parser.add_argument('--gamma_bn_momentum', type=float, default=0.5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--dataset', type=str, default='kitti', help='kitti/nuscenes')
    parser.add_argument('--pretrain_model', type=str, default='')
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--use_wandb', type=int, default=1)

    return parser.parse_args()

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_dataset = KittiInterpolationDataset(args.root, args.npoints, 5, True, True)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    net = PointINet(args.freeze).cuda()
    if args.use_wandb:
        wandb.watch(net)
    
    net.flow.load_state_dict(torch.load(args.pretrain_model))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.init_lr)
    lr_scheduler = ClippedStepLR(optimizer, args.step_size_lr, args.min_lr, args.gamma_lr)
    def update_bn_momentum(epoch):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = max(args.init_bn_momentum * args.gamma_bn_momentum ** (epoch // args.step_size_bn_momentum), args.min_bn_momentum)
    
    best_train_loss = float('inf')

    for epoch in range(args.epochs):
        update_bn_momentum(epoch)
        net.train()
        count = 0
        total_loss = 0

        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            ini_pc, mid_pc, end_pc, ini_color, mid_color, end_color, t = data
            ini_pc = ini_pc.cuda(non_blocking=True)
            mid_pc = mid_pc.cuda(non_blocking=True)
            end_pc = end_pc.cuda(non_blocking=True)
            ini_color = ini_color.cuda(non_blocking=True)
            mid_color = mid_color.cuda(non_blocking=True)
            end_color = end_color.cuda(non_blocking=True)
            t = t.cuda().float()

            optimizer.zero_grad()
            pred_mid_pc = net(ini_pc, end_pc, ini_color, end_color, t)

            loss = chamfer_loss(pred_mid_pc[:,:3,:], mid_pc[:,:3,:])

            loss.backward()
            optimizer.step()

            count += 1
            total_loss += loss.item()
            if i % 10 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item()
                ))
            
        lr_scheduler.step()
        total_loss = total_loss/count
        if args.use_wandb == 1:
            wandb.log({"loss":total_loss})
        
        print('Epoch ', epoch+1, 'finished ', 'loss = ', total_loss)

        if total_loss < best_train_loss:
            torch.save(net.state_dict(), args.save_dir+'best_train.pth')
            best_train_loss = total_loss

        print('Best train loss: {:.4f}'.format(best_train_loss))

if __name__ == '__main__':
    args = parse_args()
    if args.use_wandb == 1:
        import wandb
        wandb.init(config=args, project='PointINet')
    
    train(args)