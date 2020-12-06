import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# We utilize kaolin to implement layers for FlowNet3D
# website: https://github.com/NVIDIAGameWorks/kaolin
import kaolin as kal
from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import fps_gather_by_index
from kaolin.models.PointNet2 import ball_query
from kaolin.models.PointNet2 import group_gather_by_index
from kaolin.models.PointNet2 import three_nn
from kaolin.models.PointNet2 import three_interpolate

# We utilize pytorch3d to implement k-nearest-neighbor search
# website: https://github.com/facebookresearch/pytorch3d
from pytorch3d.ops import knn_points, knn_gather

from .utils import pdist2squared

'''
Layers for FlowNet3D
'''
class Sample(nn.Module):
    '''
    Furthest point sample
    '''
    def __init__(self, num_points):
        super(Sample, self).__init__()
        
        self.num_points = num_points
        
    def forward(self, points):
        new_points_ind = furthest_point_sampling(points.permute(0, 2, 1).contiguous(), self.num_points)
        new_points = fps_gather_by_index(points, new_points_ind)
        return new_points

class Group(nn.Module):
    '''
    kNN group for FlowNet3D
    '''
    def __init__(self, radius, num_samples, knn=False):
        super(Group, self).__init__()
        
        self.radius = radius
        self.num_samples = num_samples
        self.knn = knn
        
    def forward(self, points, new_points, features):
        if self.knn:
            dist = pdist2squared(points, new_points)
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()
        else:
            ind = ball_query(self.radius, self.num_samples, points.permute(0, 2, 1).contiguous(),
                             new_points.permute(0, 2, 1).contiguous(), False)
        grouped_points = group_gather_by_index(points, ind)
        grouped_points_new = grouped_points - new_points.unsqueeze(3)
        grouped_features = group_gather_by_index(features, ind)
        new_features = torch.cat([grouped_points_new, grouped_features], dim=1)
        return new_features

class SetConv(nn.Module):
    def __init__(self, num_points, radius, num_samples, in_channels, out_channels):
        super(SetConv, self).__init__()
        
        self.sample = Sample(num_points)
        self.group = Group(radius, num_samples)
        
        layers = []
        out_channels = [in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points, features):
        new_points = self.sample(points)
        new_features = self.group(points, new_points, features)
        new_features = self.conv(new_features)
        new_features = new_features.max(dim=3)[0]
        return new_points, new_features

class FlowEmbedding(nn.Module):
    def __init__(self, num_samples, in_channels, out_channels):
        super(FlowEmbedding, self).__init__()
        
        self.num_samples = num_samples
        
        self.group = Group(None, self.num_samples, knn=True)
        
        layers = []
        out_channels = [2*in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        new_features = self.group(points2, points1, features2)
        new_features = torch.cat([new_features, features1.unsqueeze(3).expand(-1, -1, -1, self.num_samples)], dim=1)
        new_features = self.conv(new_features)
        new_features = new_features.max(dim=3)[0]
        return new_features

class SetUpConv(nn.Module):
    def __init__(self, num_samples, in_channels1, in_channels2, out_channels1, out_channels2):
        super(SetUpConv, self).__init__()
        
        self.group = Group(None, num_samples, knn=True)
        
        layers = []
        out_channels1 = [in_channels1+3, *out_channels1]
        for i in range(1, len(out_channels1)):
            layers += [nn.Conv2d(out_channels1[i - 1], out_channels1[i], 1, bias=True), nn.BatchNorm2d(out_channels1[i], eps=0.001), nn.ReLU()]
        self.conv1 = nn.Sequential(*layers)
        
        layers = []
        if len(out_channels1) == 1:
            out_channels2 = [in_channels1+in_channels2+3, *out_channels2]
        else:
            out_channels2 = [out_channels1[-1]+in_channels2, *out_channels2]
        for i in range(1, len(out_channels2)):
            layers += [nn.Conv2d(out_channels2[i - 1], out_channels2[i], 1, bias=True), nn.BatchNorm2d(out_channels2[i], eps=0.001), nn.ReLU()]
        self.conv2 = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        new_features = self.group(points1, points2, features1)
        new_features = self.conv1(new_features)
        new_features = new_features.max(dim=3)[0]
        new_features = torch.cat([new_features, features2], dim=1)
        new_features = new_features.unsqueeze(3)
        new_features = self.conv2(new_features)
        new_features = new_features.squeeze(3)
        return new_features

class FeaturePropagation(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeaturePropagation, self).__init__()
        
        layers = []
        out_channels = [in_channels1+in_channels2, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        dist, ind = three_nn(points2.permute(0, 2, 1).contiguous(), points1.permute(0, 2, 1).contiguous())
        dist = dist * dist
        dist[dist < 1e-10] = 1e-10
        inverse_dist = 1.0 / dist
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        new_features = torch.sum(group_gather_by_index(features1, ind) * weights.unsqueeze(1), dim = 3)
        new_features = torch.cat([new_features, features2], dim=1)
        new_features = self.conv(new_features.unsqueeze(3)).squeeze(3)
        return new_features

class PointsFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        
        self.conv = nn.Sequential(*layers)
    
    def knn_group(self, points1, points2, features2, k):
        '''
        For each point in points1, query kNN points/features in points2/features2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features2: [B,C,N]
        Output:
            new_features: [B,4,N]
            nn: [B,3,N]
            grouped_features: [B,C,N]
        '''
        points1 = points1.permute(0,2,1).contiguous()
        points2 = points2.permute(0,2,1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        grouped_features = knn_gather(features2.permute(0,2,1), nn_idx)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)

        return new_features.permute(0,3,1,2).contiguous(),\
            nn.permute(0,3,1,2).contiguous(),\
            grouped_features.permute(0,3,1,2).contiguous()
    
    def forward(self, points1, points2, features1, features2, k, t):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,C,N] (only for inference of additional features)
            features2: [B,C,N] (only for inference of additional features)
            k: int, number of kNN cluster
            t: [B], time step in (0,1)
        Output:
            fused_points: [B,3+C,N]
        '''
        N = points1.shape[-1]
        B = points1.shape[0] # batch size

        new_features_list = []
        new_grouped_points_list = []
        new_grouped_features_list = []

        for i in range(B):
            t1 = t[i]
            new_points1 = points1[i:i+1,:,:]
            new_points2 = points2[i:i+1,:,:]
            new_features1 = features1[i:i+1,:,:]
            new_features2 = features2[i:i+1,:,:]

            N2 = int(N*t1)
            N1 = N - N2

            k2 = int(k*t1)
            k1 = k - k2

            randidx1 = torch.randperm(N)[:N1]
            randidx2 = torch.randperm(N)[:N2]
            new_points = torch.cat((new_points1[:,:,randidx1], new_points2[:,:,randidx2]), dim=-1)

            new_features1, grouped_points1, grouped_features1 = self.knn_group(new_points, new_points1, new_features1, k1)
            new_features2, grouped_points2, grouped_features2 = self.knn_group(new_points, new_points2, new_features2, k2)

            new_features = torch.cat((new_features1, new_features2), dim=-1)
            new_grouped_points = torch.cat((grouped_points1, grouped_points2), dim=-1)
            new_grouped_features = torch.cat((grouped_features1, grouped_features2), dim=-1)

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)
            new_grouped_features_list.append(new_grouped_features)

        new_features = torch.cat(new_features_list, dim=0)
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)
        new_grouped_features = torch.cat(new_grouped_features_list, dim=0)

        new_features = self.conv(new_features)
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]
        weights = F.softmax(new_features, dim=-1)

        C = features1.shape[1]
        weights = weights.unsqueeze(1).repeat(1,3+C,1,1)
        fused_points = torch.cat([new_grouped_points, new_grouped_features], dim=1)
        fused_points = torch.sum(torch.mul(weights, fused_points), dim=-1, keepdim=False)

        return fused_points

def knn_group_withI(points1, points2, intensity2, k):
    '''
    Input:
        points1: [B,3,N]
        points2: [B,3,N]
        intensity2: [B,1,N]
    '''
    points1 = points1.permute(0,2,1).contiguous()
    points2 = points2.permute(0,2,1).contiguous()
    _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
    points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1) # [B,M,k,3]
    grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
    grouped_features = knn_gather(intensity2.permute(0,2,1), nn_idx) # [B,M,k,1]
    new_features = torch.cat([points_resi, grouped_dist], dim=-1)
    
    # [B,5,M,k], [B,3,M,k], [B,1,M,k]
    return new_features.permute(0,3,1,2).contiguous(), \
        nn.permute(0,3,1,2).contiguous(), \
        grouped_features.permute(0,3,1,2).contiguous()