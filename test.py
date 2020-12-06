import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.models import PointINet
from models.utils import chamfer_loss, EMD

from data.interpolation_data import NuscenesDataset, KittiInterpolationDataset

import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--pretrain_model', type=str, default='./pretrain_model/interp_kitti.pth')
    parser.add_argument('--pretrain_flow_model', type=str, default='./pretrain_model/flownet3d_kitti_odometry_maxbias5.pth')
    parser.add_argument('--dataset', type=str, default='kitti', help='kitti/nuscenes')
    parser.add_argument('--scenelist', type=str, default='')

    return parser.parse_args()

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset == 'nuscenes':
        scene_list = args.scenelist
        test_set = NuscenesDataset(args.npoints, args.root, scene_list, True, 5, False)
    elif args.dataset == 'kitti':
        test_set = KittiInterpolationDataset(args.root, args.npoints, 5, False, True)
    
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True)
    
    net = PointINet(freeze=1)
    net.load_state_dict(torch.load(args.pretrain_model))
    net.flow.load_state_dict(torch.load(args.pretrain_flow_model))
    net.eval()
    net.cuda()

    with torch.no_grad():

        chamfer_loss_list = []
        emd_loss_list = []

        pbar = tqdm(test_loader)
        for data in pbar:

            ini_pc, mid_pc, end_pc, ini_color, mid_color, end_color, t = data
            ini_pc = ini_pc.cuda(non_blocking=True)
            mid_pc = mid_pc.cuda(non_blocking=True)
            end_pc = end_pc.cuda(non_blocking=True)
            ini_color = ini_color.cuda(non_blocking=True)
            mid_color = mid_color.cuda(non_blocking=True)
            end_color = end_color.cuda(non_blocking=True)
            t = t.cuda().float()
            
            pred_mid_pc = net(ini_pc, end_pc, ini_color, end_color, t)

            cd = chamfer_loss(pred_mid_pc[:,:3,:], mid_pc[:,:3,:])
            emd = EMD(pred_mid_pc[:,:3,:], mid_pc[:,:3,:])

            cd = cd.squeeze().cpu().numpy()
            emd = emd.squeeze().cpu().numpy()

            chamfer_loss_list.append(cd)
            emd_loss_list.append(emd)

            pbar.set_description('CD:{:.3} EMD:{:.3}'.format(cd, emd))

    chamfer_loss_array = np.array(chamfer_loss_list)
    emd_loss_array = np.array(emd_loss_list)
    mean_chamfer_loss = np.mean(chamfer_loss_array)
    mean_emd_loss = np.mean(emd_loss_array)

    print("Mean chamfer distance: ", mean_chamfer_loss)
    print("Mean earth mover's distance: ", mean_emd_loss)

if __name__ == '__main__':
    args = parse_args()
    test(args)