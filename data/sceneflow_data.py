import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from tqdm import tqdm

class Flythings3D(Dataset):
    def __init__(self, npoints=2048, root='data/data_processed_maxcut_35_20k_2k_8192', train=True, cache=None):
        self.npoints = npoints
        self.train = train
        self.root = root
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        
        if cache is None:
            self.cache = {}
        else:
            self.cache = cache
        
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype('float32')
                pos2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32') / 255
                color2 = data['color2'].astype('float32') / 255
                flow = data['flow'].astype('float32')
                mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center
        
        pos1 = torch.from_numpy(pos1).t()
        pos2 = torch.from_numpy(pos2).t()
        color1 = torch.from_numpy(color1).t()
        color2 = torch.from_numpy(color2).t()
        flow = torch.from_numpy(flow).t()
        mask1 = torch.from_numpy(mask1)

        return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)

class KittiSceneFlowDataset(Dataset):
    def __init__(self, root, npoints, train=True):
        self.npoints = npoints
        self.root = root
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
    
    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pos1 = data['pos1'].astype('float32')
            pos2 = data['pos2'].astype('float32')
            flow = data['gt'].astype('float32')

            n1 = pos1.shape[0]
            n2 = pos2.shape[0]

            if n1 >= self.npoints:
                sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
            if n2 >= self.npoints:
                sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)
            
            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            flow = flow[sample_idx1, :]

            color1 = np.zeros([self.npoints,3], dtype=np.float32)
            color2 = np.zeros([self.npoints,3], dtype=np.float32)
            mask1 = np.ones([self.npoints])

        pos1 = torch.from_numpy(pos1).t()
        pos2 = torch.from_numpy(pos2).t()
        color1 = torch.from_numpy(color1).t()
        color2 = torch.from_numpy(color2).t()
        flow = torch.from_numpy(flow).t()
        mask1 = torch.from_numpy(mask1)

        return pos1, pos2, color1, color2, flow, mask1
    
    def __len__(self):
        return len(self.datapath)

class KittiOdometrySceneflow(Dataset):
    def __init__(self, root, npoints, max_bias, train=True):
        self.npoints = npoints
        self.root = root
        self.train = train
        self.max_bias = max_bias
        self.datapath = glob.glob(os.path.join(self.root, '*.bin'))
        self.datapath = sorted(self.datapath)
    
    def get_cloud(self, fn):
        pc = np.fromfile(fn, dtype=np.float32, count=-1).reshape([-1,4])
        return pc
    
    def __getitem__(self, index):
        pc1_fn = os.path.join(self.root, self.datapath[index])
        max_ind = len(self.datapath)
        if index <= self.max_bias:
            bias = np.random.randint(1, self.max_bias+1)
        elif index >= max_ind - self.max_bias:
            bias = np.random.randint(-self.max_bias,0)
        else:
            bias = np.random.randint(-self.max_bias,self.max_bias+1)
            if bias == 0:
                bias = bias + 1
        
        pc2_fn = os.path.join(self.root, self.datapath[index+bias])

        points1 = self.get_cloud(pc1_fn)
        points2 = self.get_cloud(pc2_fn)

        n1 = points1.shape[0]
        n2 = points2.shape[0]

        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)
        
        points1 = points1[sample_idx1, :3].astype('float32')
        points2 = points2[sample_idx2, :3].astype('float32')
        color1 = np.zeros([self.npoints,3]).astype('float32')
        color2 = np.zeros([self.npoints,3]).astype('float32')

        points1 = torch.from_numpy(points1).t()
        points2 = torch.from_numpy(points2).t()
        color1 = torch.from_numpy(color1).t()
        color2 = torch.from_numpy(color2).t()

        return points1, points2, color1, color2
    
    def __len__(self):
        return len(self.datapath)

class NuScenesFlow(Dataset):
    def __init__(self, root, npoints, scenes_list, max_bias):
        self.npoints = npoints
        self.root = root
        self.scenes = self.read_scene_list(scenes_list)
        self.times_list, self.fns_list = self.load_scene(self.scenes)
        self.max_bias = max_bias
        self.dataset_fns, self.dataset_times = self.make_dataset()
    
    def read_scene_list(self, scenes_list):
        scenes = []
        with open(scenes_list, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                scenes.append(line)
        return scenes
    
    def load_scene(self, scenes):
        times_list = []
        fns_list = []
        
        for scene in scenes:
            scene_file = os.path.join('./data/scene-split', scene+'.txt')
            times = []
            fns = []
            with open(scene_file) as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    fn = line[0]
                    timestamp = float(line[1])
                    fns.append(fn)
                    times.append(timestamp)
            times_list.append(times)
            fns_list.append(fns)
        return times_list, fns_list
    
    def make_dataset(self):
        fns_lists = []
        times_lists = []
        for i in tqdm(range(len(self.times_list))):
            times = self.times_list[i]
            fns = self.fns_list[i]
            max_ind = len(times)
            ini_index = 0
            while (ini_index < max_ind - self.max_bias):
                if ini_index <= self.max_bias:
                    bias = np.random.randint(1, self.max_bias+1)
                else:
                    bias = np.random.randint(-self.max_bias,self.max_bias+1)
                    if bias == 0:
                        bias = bias + 1
                end_index = ini_index + bias
                fns_lists.append([fns[ini_index], fns[end_index]])
                times_lists.append([times[ini_index], times[end_index]])
                ini_index += 1
        return fns_lists, times_lists
    
    def get_lidar(self, fn):
        scan_in = np.fromfile(fn, np.float32).reshape(-1,5)
        scan = np.zeros((scan_in.shape[0],4),np.float32)
        scan = scan_in[:,:4]
        return scan
    
    def __getitem__(self, index):
        pc1_fn = os.path.join(self.root, 'sweeps', 'LIDAR_TOP', self.dataset_fns[index][0])
        pc2_fn = os.path.join(self.root, 'sweeps', 'LIDAR_TOP', self.dataset_fns[index][1])
        points1 = self.get_lidar(pc1_fn)
        points2 = self.get_lidar(pc2_fn)

        n1 = points1.shape[0]
        n2 = points2.shape[0]

        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)
        
        points1 = points1[sample_idx1, :3].astype('float32')
        points2 = points2[sample_idx2, :3].astype('float32')
        color1 = np.zeros([self.npoints,3]).astype('float32')
        color2 = np.zeros([self.npoints,3]).astype('float32')

        points1 = torch.from_numpy(points1).t()
        points2 = torch.from_numpy(points2).t()
        color1 = torch.from_numpy(color1).t()
        color2 = torch.from_numpy(color2).t()

        return points1, points2, color1, color2
    
    def __len__(self):
        return len(self.dataset_fns)