import torch
import torch.nn as nn
import numpy as np

from models.models import PointINet

import mayavi.mlab as mlab

import argparse
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--npoints', type=int, default=32768)
    parser.add_argument('--pretrain_model', type=str, default='./pretrain_model/interp_kitti.pth')
    parser.add_argument('--pretrain_flow_model', type=str, default='./pretrain_model/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--is_save', type=int, default=1)
    parser.add_argument('--visualize', type=int, default=1)

    return parser.parse_args()

def get_lidar(fn, npoints):
    points = np.fromfile(fn, dtype=np.float32, count=-1).reshape([-1,4])
    raw_num = points.shape[0]
    if raw_num >= npoints:
        sample_idx = np.random.choice(raw_num, npoints, replace=False)
    else:
        sample_idx = np.concatenate((np.arange(raw_num), np.random.choice(raw_num, npoints - raw_num, replace=True)), axis=-1)
    
    pc = points[sample_idx, :]
    pc = torch.from_numpy(pc).t()
    color = np.zeros([npoints,3]).astype('float32')
    color = torch.from_numpy(color).t()

    pc = pc.unsqueeze(0).cuda()
    color = color.unsqueeze(0).cuda()

    return pc, color

def demo(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fn1 = './data/demo_data/original/000000.bin'
    fn2 = './data/demo_data/original/000001.bin'

    net = PointINet()
    net.load_state_dict(torch.load(args.pretrain_model))
    net.flow.load_state_dict(torch.load(args.pretrain_flow_model))
    net.eval()
    net.cuda()

    interp_scale = 5
    t_array = np.arange(1.0/interp_scale, 1.0, 1.0/interp_scale, dtype=np.float32)

    with torch.no_grad():
        pc1, color1 = get_lidar(fn1, args.npoints)
        pc2, color2 = get_lidar(fn2, args.npoints)

        for i in range(interp_scale-1):
            t = t_array[i]
            t = torch.tensor([t])
            t = t.cuda().float()

            pred_mid_pc = net(pc1, pc2, color1, color2, t)

            ini_pc = pc1.squeeze(0).permute(1,0).cpu().numpy()
            end_pc = pc2.squeeze(0).permute(1,0).cpu().numpy()

            
            pred_mid_pc = pred_mid_pc.squeeze(0).permute(1,0).cpu().numpy()

            if args.visualize == 1:
                fig = mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=(0,0,0), engine=None, size=(1600, 1000))
                mlab.points3d(ini_pc[:,0],ini_pc[:,1],ini_pc[:,2],color=(0,0,1),scale_factor=0.2,figure=fig, mode='sphere')
                mlab.points3d(end_pc[:,0],end_pc[:,1],end_pc[:,2],color=(0,1,0),scale_factor=0.2,figure=fig, mode='sphere')
                mlab.points3d(pred_mid_pc[:,0],pred_mid_pc[:,1],pred_mid_pc[:,2],color=(1,0,0),scale_factor=0.2,figure=fig, mode='sphere')
                mlab.show()
            
            if args.is_save == 1:
                save_dir = './data/demo_data/interpolated'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = os.path.join(save_dir, str(t.squeeze().cpu().numpy())+'.bin')
                pred_mid_pc.tofile(save_name)
                print("save interpolated point clouds to:", save_name)

if __name__ == '__main__':
    args = parse_args()
    demo(args)