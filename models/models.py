import torch
import torch.nn as nn
import datetime
import numpy as np

from .layers import SetConv, FlowEmbedding, SetUpConv, FeaturePropagation, PointsFusion

class FlowNet3D(nn.Module):
    '''
    Implementation of FlowNet3D (CVPR 2019) in PyTorch
    We refer to original Tensorflow implementation (https://github.com/xingyul/flownet3d) 
    and open source PyTorch implementation (https://github.com/multimodallearning/flownet3d.pytorch)
    to implement the code for FlowNet3D.
    '''
    def __init__(self):
        super(FlowNet3D, self).__init__()

        self.set_conv1 = SetConv(1024, 0.5, 16, 3, [32, 32, 64])
        self.set_conv2 = SetConv(256, 1.0, 16, 64, [64, 64, 128])
        self.flow_embedding = FlowEmbedding(64, 128, [128, 128, 128])
        self.set_conv3 = SetConv(64, 2.0, 8, 128, [128, 128, 256])
        self.set_conv4 = SetConv(16, 4.0, 8, 256, [256, 256, 512])
        self.set_upconv1 = SetUpConv(8, 512, 256, [], [256, 256])
        self.set_upconv2 = SetUpConv(8, 256, 256, [128, 128, 256], [256])
        self.set_upconv3 = SetUpConv(8, 256, 64, [128, 128, 256], [256])
        self.fp = FeaturePropagation(256, 3, [256, 256])
        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )
    
    def forward(self, points1, points2, features1, features2):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,3,N] (colors for Flythings3D and zero for LiDAR)
            features2: [B,3,N] (colors for Flythings3D and zero for LiDAR)
        Output:
            flow: [B,3,N]
        '''
        points1_1, features1_1 = self.set_conv1(points1, features1)
        points1_2, features1_2 = self.set_conv2(points1_1, features1_1)

        points2_1, features2_1 = self.set_conv1(points2, features2)
        points2_2, features2_2 = self.set_conv2(points2_1, features2_1)

        embedding = self.flow_embedding(points1_2, points2_2, features1_2, features2_2)

        points1_3, features1_3 = self.set_conv3(points1_2, embedding)
        points1_4, features1_4 = self.set_conv4(points1_3, features1_3)

        new_features1_3 = self.set_upconv1(points1_4, points1_3, features1_4, features1_3)
        new_features1_2 = self.set_upconv2(points1_3, points1_2, new_features1_3, torch.cat([features1_2, embedding], dim=1))
        new_features1_1 = self.set_upconv3(points1_2, points1_1, new_features1_2, features1_1)
        new_features1 = self.fp(points1_1, points1, new_features1_1, features1)

        flow = self.classifier(new_features1)

        return flow

class PointINet(nn.Module):
    def __init__(self, freeze=1):
        super(PointINet, self).__init__()
        self.flow = FlowNet3D()
        if freeze == 1:
            for p in self.parameters():
                p.requires_grad = False

        self.fusion = PointsFusion(4, [64, 64, 128])
    
    def forward(self, points1, points2, features1, features2, t):
        '''
        Input:
            points1: [B,3+C,N]
            points2: [B,3+C,N]
            features1: [B,3,N] (zeros)
            features2: [B,3,N]
        '''

        points1_feature = points1[:,3:,:].contiguous()
        points2_feature = points2[:,3:,:].contiguous()
        points1 = points1[:,:3,:].contiguous()
        points2 = points2[:,:3,:].contiguous()
        

        # Estimate 3D scene flow
        with torch.no_grad():
            flow_forward = self.flow(points1, points2, features1, features2)
            flow_backward = self.flow(points2, points1, features2, features1)
        
        t = t.unsqueeze(1).unsqueeze(1)

        # Warp two input point clouds
        warped_points1_xyz = points1 + flow_forward * t
        warped_points2_xyz = points2 + flow_backward * (1-t)

        k = 32

        # Points fusion
        fused_points = self.fusion(warped_points1_xyz, warped_points2_xyz, points1_feature, points2_feature, k, t)

        return fused_points

if __name__ == '__main__':
    net = PointINet(freeze=1).cuda()

    points1 = torch.rand(1,4,16384).cuda()
    points2 = torch.rand(1,4,16384).cuda()
    features1 = torch.rand(1,3,16384).cuda()
    features2 = torch.rand(1,3,16384).cuda()

    t = np.array([0.2]).astype('float32')
    t = torch.from_numpy(t).cuda()

    fused_points = net(points1, points2, features1, features2, t)

    print(fused_points.shape)