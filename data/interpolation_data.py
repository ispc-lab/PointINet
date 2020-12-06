import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import numpy as np
import os
from tqdm import tqdm

class KittiInterpolationDataset(Dataset):
    def __init__(self, root, npoints, interval, train=True, use_intensity=True):
        super(KittiInterpolationDataset, self).__init__()
        self.root = root
        self.npoints = npoints
        self.dataroot = os.path.join(self.root, 'velodyne')
        self.timeroot = os.path.join(self.root, 'times.txt')
        self.train = train
        self.use_intensity = use_intensity
        self.times = []
        self.read_times()
        self.datapath = glob.glob(os.path.join(self.dataroot, '*.bin'))
        self.datapath = sorted(self.datapath)
        self.interval = interval
        self.dataset = self.make_dataset()
    
    def read_times(self):
        with open(self.timeroot) as f:
            for line in f.readlines():
                line = line.strip()
                time = float(line)
                self.times.append(time)
    
    def get_cloud(self, file_name):
        pc = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1,4])
        return pc
    
    def make_dataset(self):
        max_ind = len(self.datapath)
        ini_index = 0
        end_index = 0
        index_lists = []
        while (ini_index < max_ind - self.interval):
            end_index = ini_index + self.interval
            if self.train:
                mid_index = np.random.randint(1,self.interval) + ini_index
                triple = [ini_index, mid_index, end_index]
                index_lists.append(triple)
            else:
                for bias in range(1, self.interval):
                    mid_index = bias + ini_index
                    triple = [ini_index, mid_index, end_index]
                    index_lists.append(triple)
            ini_index = end_index
        return index_lists

    def __getitem__(self, index):
        ini_index, mid_index, end_index = self.dataset[index]
        ini_pc = self.get_cloud(os.path.join(self.dataroot, self.datapath[ini_index]))
        mid_pc = self.get_cloud(os.path.join(self.dataroot, self.datapath[mid_index]))
        end_pc = self.get_cloud(os.path.join(self.dataroot, self.datapath[end_index]))

        ini_n = ini_pc.shape[0]
        mid_n = mid_pc.shape[0]
        end_n = end_pc.shape[0]

        if ini_n >= self.npoints:
            sample_idx_ini = np.random.choice(ini_n, self.npoints, replace=False)
        else:
            sample_idx_ini = np.concatenate((np.arange(ini_n), np.random.choice(ini_n, self.npoints - ini_n, replace=True)), axis=-1)
        if mid_n >= self.npoints:
            sample_idx_mid = np.random.choice(mid_n, self.npoints, replace=False)
        else:
            sample_idx_mid = np.concatenate((np.arange(mid_n), np.random.choice(mid_n, self.npoints - mid_n, replace=True)), axis=-1)
        if end_n >= self.npoints:
            sample_idx_end = np.random.choice(end_n, self.npoints, replace=False)
        else:
            sample_idx_end = np.concatenate((np.arange(end_n), np.random.choice(end_n, self.npoints - end_n, replace=True)), axis=-1)
        
        if self.use_intensity:
            ini_pc = ini_pc[sample_idx_ini, :].astype('float32')
            mid_pc = mid_pc[sample_idx_mid, :].astype('float32')
            end_pc = end_pc[sample_idx_end, :].astype('float32')
        else:
            ini_pc = ini_pc[sample_idx_ini, :3].astype('float32')
            mid_pc = mid_pc[sample_idx_mid, :3].astype('float32')
            end_pc = end_pc[sample_idx_end, :3].astype('float32')

        ini_color = np.zeros([self.npoints,3]).astype('float32')
        mid_color = np.zeros([self.npoints,3]).astype('float32')
        end_color = np.zeros([self.npoints,3]).astype('float32')

        ini_pc = torch.from_numpy(ini_pc).t()
        mid_pc = torch.from_numpy(mid_pc).t()
        end_pc = torch.from_numpy(end_pc).t()

        ini_color = torch.from_numpy(ini_color).t()
        mid_color = torch.from_numpy(mid_color).t()
        end_color = torch.from_numpy(end_color).t()
    
        ini_t = self.times[ini_index]
        mid_t = self.times[mid_index]
        end_t = self.times[end_index]

        t = (mid_t-ini_t)/(end_t-ini_t)

        return ini_pc, mid_pc, end_pc, ini_color, mid_color, end_color, t
    
    def __len__(self):
        return len(self.dataset)

class NuscenesDataset(Dataset):
    def __init__(self, npoints, root, scenes_list, use_intensity, interval, train):
        self.npoints = npoints
        self.root = root
        self.train = train
        self.scenes = self.read_scene_list(scenes_list)
        self.times_list, self.fns_list = self.load_scene(self.scenes)
        self.interval = interval
        self.dataset_fns, self.dataset_times = self.make_dataset()
        self.use_intensity = use_intensity
        
    
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
        for i in range(len(self.times_list)):
            times = self.times_list[i]
            fns = self.fns_list[i]
            max_ind = len(times)
            ini_index = 0
            end_index = 0
            while (ini_index < max_ind - self.interval):
                end_index = ini_index + self.interval
                if self.train:
                    bias = np.random.randint(1, self.interval)
                    fns_lists.append([fns[ini_index], fns[ini_index+bias], fns[end_index]])
                    times_lists.append([times[ini_index], times[ini_index+bias], times[end_index]])
                else:
                    for bias in range(1, self.interval):
                        fns_lists.append([fns[ini_index], fns[ini_index+bias], fns[end_index]])
                        times_lists.append([times[ini_index], times[ini_index+bias], times[end_index]])
                ini_index = end_index
        return fns_lists, times_lists

    def get_lidar(self, fn):
        scan_in = np.fromfile(fn, np.float32).reshape(-1,5)
        scan = np.zeros((scan_in.shape[0],4),np.float32)
        scan = scan_in[:,:4]
        return scan
    
    def __getitem__(self, index):
        ini_pc = self.get_lidar(os.path.join(self.root, 'sweeps', 'LIDAR_TOP', self.dataset_fns[index][0]))
        mid_pc = self.get_lidar(os.path.join(self.root, 'sweeps', 'LIDAR_TOP', self.dataset_fns[index][1]))
        end_pc = self.get_lidar(os.path.join(self.root, 'sweeps', 'LIDAR_TOP', self.dataset_fns[index][2]))

        ini_n = ini_pc.shape[0]
        mid_n = mid_pc.shape[0]
        end_n = end_pc.shape[0]

        if ini_n >= self.npoints:
            sample_idx_ini = np.random.choice(ini_n, self.npoints, replace=False)
        else:
            sample_idx_ini = np.concatenate((np.arange(ini_n), np.random.choice(ini_n, self.npoints - ini_n, replace=True)), axis=-1)
        if mid_n >= self.npoints:
            sample_idx_mid = np.random.choice(mid_n, self.npoints, replace=False)
        else:
            sample_idx_mid = np.concatenate((np.arange(mid_n), np.random.choice(mid_n, self.npoints - mid_n, replace=True)), axis=-1)
        if end_n >= self.npoints:
            sample_idx_end = np.random.choice(end_n, self.npoints, replace=False)
        else:
            sample_idx_end = np.concatenate((np.arange(end_n), np.random.choice(end_n, self.npoints - end_n, replace=True)), axis=-1)
        
        if self.use_intensity:
            ini_pc = ini_pc[sample_idx_ini, :].astype('float32')
            mid_pc = mid_pc[sample_idx_mid, :].astype('float32')
            end_pc = end_pc[sample_idx_end, :].astype('float32')
        else:
            ini_pc = ini_pc[sample_idx_ini, :3].astype('float32')
            mid_pc = mid_pc[sample_idx_mid, :3].astype('float32')
            end_pc = end_pc[sample_idx_end, :3].astype('float32')
        
        ini_color = np.zeros([self.npoints,3]).astype('float32')
        mid_color = np.zeros([self.npoints,3]).astype('float32')
        end_color = np.zeros([self.npoints,3]).astype('float32')

        ini_pc = torch.from_numpy(ini_pc).t()
        mid_pc = torch.from_numpy(mid_pc).t()
        end_pc = torch.from_numpy(end_pc).t()

        ini_color = torch.from_numpy(ini_color).t()
        mid_color = torch.from_numpy(mid_color).t()
        end_color = torch.from_numpy(end_color).t()

        ini_t = self.dataset_times[index][0]
        mid_t = self.dataset_times[index][1]
        end_t = self.dataset_times[index][2]

        t = (mid_t-ini_t)/(end_t-ini_t)

        return ini_pc, mid_pc, end_pc, ini_color, mid_color, end_color, t
    
    def __len__(self):
        return len(self.dataset_fns)