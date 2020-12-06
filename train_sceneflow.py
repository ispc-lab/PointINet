import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from data.sceneflow_data import Flythings3D, KittiSceneFlowDataset, KittiOdometrySceneflow, NuScenesFlow

from models.models import FlowNet3D
from models.utils import ClippedStepLR, flow_criterion, chamfer_loss

from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='FlowNet3d')

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
    parser.add_argument('--npoints', type=int, default=8192)
    parser.add_argument('--dataset', type=str, default='Flythings3D', help='Flythings3D/Kitti/nuscenes')
    parser.add_argument('--pretrain_model', type=str, default='')
    parser.add_argument('--max_bias', type=int, default=2)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--train_type', type=str, default='init', help='init/refine')

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
    if args.dataset == 'Flythings3D':
        train_dataset = Flythings3D(npoints=args.npoints, root=args.root, train=True)
    elif args.dataset == 'Kitti':
        train_dataset = KittiSceneFlowDataset(args.root, args.npoints, True)
    else:
        raise('Invalid dataset')
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    net = FlowNet3D().cuda()

    if args.use_wandb:
        wandb.watch(net)
    
    if args.dataset == 'Flythings3D':
        net.apply(init_weights)
    elif args.dataset == 'Kitti':
        net.load_state_dict(torch.load(args.pretrain_model))
    else:
        raise('Invalid dataset')
    
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
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
            points1, points2, features1, features2, flow, mask1 = data
            points1 = points1.cuda(non_blocking=True)
            points2 = points2.cuda(non_blocking=True)
            features1 = features1.cuda(non_blocking=True)
            features2 = features2.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            mask1 = mask1.cuda(non_blocking=True).float()

            optimizer.zero_grad()
            pred_flow = net(points1, points2, features1, features2)

            loss = flow_criterion(pred_flow, flow, mask1)
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
        if args.use_wandb:
            wandb.log({"loss":total_loss})
        
        print('Epoch ', epoch+1, 'finished ', 'loss = ', total_loss)
        if total_loss < best_train_loss:
            torch.save(net.state_dict(), args.save_dir+'best_train.pth')
            best_train_loss = total_loss
        
        print('Best train loss: {:.4f}'.format(best_train_loss))

def train_unsupervised(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset == 'Kitti':
        train_dataset = KittiOdometrySceneflow(root=args.root, npoints=args.npoints, max_bias=args.max_bias)
    elif args.dataset == 'nuscenes':
        scenes_list = './data/nuscenes_trainlist.txt'
        train_dataset = NuScenesFlow(root=args.root, npoints=args.npoints, scenes_list=scenes_list, max_bias=args.max_bias)

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    net = FlowNet3D().cuda()

    if args.use_wandb:
        wandb.watch(net)
    
    net.load_state_dict(torch.load(args.pretrain_model))

    optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
    lr_scheduler = ClippedStepLR(optimizer, args.step_size_lr, args.min_lr, args.gamma_lr)

    def update_bn_momentum(epoch):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = max(args.init_bn_momentum * args.gamma_bn_momentum ** (epoch // args.step_size_bn_momentum), args.min_bn_momentum)

    best_train_loss = float('inf')
    
    for epoch in range(args.epochs):
        net.train()
        count = 0

        total_loss = 0

        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            points1, points2, features1, features2 = data
            points1 = points1.cuda(non_blocking=True)
            points2 = points2.cuda(non_blocking=True)
            features1 = features1.cuda(non_blocking=True)
            features2 = features2.cuda(non_blocking=True)

            optimizer.zero_grad()
            pred_flow = net(points1, points2, features1, features2)
            trans_points1 = points1 + pred_flow

            loss = chamfer_loss(trans_points1, points2)

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
    
    if args.train_type == 'init':
        train(args)
    elif args.train_type == 'refine':
        train_unsupervised(args)