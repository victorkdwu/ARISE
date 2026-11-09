import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter
import time

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class Magnifier(nn.Module):
    def __init__(self, input_size=16):
        super(Magnifier, self).__init__()

        self.input_size = input_size
        self.thr = 1
        d_dim = 20
        self._stage_out_channels = 80
        self.flatten_size = int(math.ceil(self.input_size/(3*5))*self._stage_out_channels)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=d_dim, kernel_size=3,  # 20
                      stride=1, dilation=1, padding=1, bias=False, groups=1),
            nn.BatchNorm1d(d_dim),
            hswish()
        )
        output_channels = 2*d_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d_dim, out_channels=d_dim, kernel_size=3,  # 20
                      stride=1, dilation=2, padding=2, bias=False, groups=d_dim),
            nn.BatchNorm1d(d_dim),
            nn.Conv1d(in_channels=d_dim, out_channels=output_channels, kernel_size=1,  # 20
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(output_channels),
            hswish()
        )
        input_channels = output_channels
        output_channels = self._stage_out_channels
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,  # 20
                      stride=1, dilation=3, padding=3, bias=False, groups=input_channels),
            nn.BatchNorm1d(input_channels),
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=1,  # 20
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(output_channels),
            hswish()
        )
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)  # upper(20/3)=7
        input_channels = output_channels
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,
                      stride=1, dilation=1, padding=1, bias=False, groups=1),  # 7
            nn.BatchNorm1d(output_channels),
            hswish(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_size, self.input_size), # 7*80=560
            nn.BatchNorm1d(1),
            hswish()
        )
        self.flatten = nn.Sequential(
            nn.Flatten()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = self.flatten(out)
        out = torch.unsqueeze(out, dim=1)
        out = self.decoder(out)
        out = torch.reshape(out, (-1,5,int(self.input_size/5)))
        return out
    
    def encode(self, x):
        out = self.conv(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = self.flatten(out)
        code = torch.unsqueeze(out, dim=1)
        out = torch.unsqueeze(out, dim=1)
        out = self.decoder(out)
        out = torch.reshape(out, (-1,5,int(self.input_size/5)))
        return code.detach().cpu().numpy(), out

    def forward_latent(self, x):
        """(latent code, reconstruction) with gradients; the code is what encode() returns."""
        out = self.conv(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        z = self.flatten(out)
        out = self.decoder(torch.unsqueeze(z, dim=1))
        return z, torch.reshape(out, (-1,5,int(self.input_size/5)))

    def rmse_t(self, output, target):
        """Per-row RMSE as a tensor (same value as excute_RMSE, keeps gradients)."""
        diff = torch.reshape(output - target, (-1, self.input_size))
        return torch.sqrt(diff.pow(2).sum(1) / self.input_size)

    def excute_RMSE(self, input, target):
        input = torch.reshape(input, (-1,1,self.input_size))
        target = torch.reshape(target, (-1,1,self.input_size))
        mse = (input - target).pow(2).sum(2) / self.input_size
        rmse = torch.sqrt(mse)
        return rmse.detach().cpu().numpy().reshape(-1)  # 0 :error 1:normal

    def pred(self, input, target):
        rmse = self.excute_RMSE(input, target)
        return rmse > self.thr

    def set_thr(self, thr):
        self.thr = thr
        
        


class EmbeddingAE(nn.Module):
    def __init__(self, input_size=105, hidden_size=32, pred_k=7, pred_margin=1.0):
        super(EmbeddingAE, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pred_k = pred_k
        self.pred_margin = pred_margin
        
        self.flatten = nn.Sequential(
            nn.Flatten()
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size*2),
            nn.BatchNorm1d(1),
            hswish(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.BatchNorm1d(1),
            hswish()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = torch.unsqueeze(x, dim=1)
        hidden = self.encoder(x).squeeze()
        return hidden
    
    def pred(self, inputs, hidden_anchors, y_anchors):
        hidden_inputs = self.forward(inputs)
        # hidden_anchors = self.forward(X_anchors)

        distances = torch.cdist(hidden_inputs, hidden_anchors)
        
        topk_indices = torch.argmin(distances, dim=1).reshape(-1,1)
        label = y_anchors[topk_indices]
        topk_indices = torch.topk(distances, k=self.pred_k, dim=1, largest=False).indices

        votes = y_anchors[topk_indices].cpu().numpy()
        label = [Counter(row).most_common(1)[0][0] for row in votes]
        
        # return
        
        return label, topk_indices.squeeze().cpu().numpy().tolist()
    
    # def pred(self, inputs, X_anchors, y_anchors):
    #     hidden_inputs = self.forward(inputs)
    #     hidden_anchors = self.forward(X_anchors)
        
    #     distances = torch.cdist(hidden_inputs, hidden_anchors)
    #     # 计算余弦相似度（范围[-1,1]）
    #     # similarities = torch.mm(hidden_inputs, hidden_anchors.T)  # [B, L]
    #     margin_mask = (distances <= self.pred_margin).view(distances.size(0), -1)  # [B, L]
    #     # topk_indices = torch.topk(distances, k=self.pred_k, dim=1, largest=False).indices
        
    #     print(margin_mask)
        
    #     # votes = y_anchors[topk_indices].cpu().numpy()
    #     votes = y_anchors[margin_mask].cpu().numpy()
    #     label = [Counter(row).most_common(1)[0][0] for row in votes]
        
    #     return label, margin_mask.squeeze().cpu().numpy().tolist()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
    
class EmbeddingLoss(nn.Module):
    def __init__(self, margin=0.5, nearest_beta=0.1):
        super(EmbeddingLoss, self).__init__()
        self.margin = margin
        self.nearest_beta = nearest_beta

    def forward(self, X_inputs, y_inputs, X_anchors, y_anchors):
        # print(X_inputs.shape)
        # print(y_inputs.shape)
        # print(X_anchors.shape)
        # print(y_anchors.shape)
        
        X_inputs = F.normalize(X_inputs, p=2, dim=1)  # [B, D]
        X_anchors = F.normalize(X_anchors, p=2, dim=1)  # [L, D]
        distances = torch.cdist(X_inputs.unsqueeze(0), X_anchors.unsqueeze(0)).squeeze()
        # distances_same_label = distances[torch.nonzero(y_inputs == y_anhcors)].squeeze(1)
        # distances_diff_label = distances[torch.nonzero(y_inputs != y_anhcors)].squeeze(1)
        
        # 分离同类与异类距离
        same_mask = (y_inputs.unsqueeze(1) == y_anchors.unsqueeze(0))  # [B, L]
        diff_mask = ~same_mask
        
        # print(same_mask.shape)
        
        # 同类距离计算
        distances_same = distances[same_mask].view(distances.size(0), -1)  # [B, K]
        min_same_dist, _ = distances_same.min(dim=1)  # 同类最小距离 [B]
        mean_same_dist = distances_same.mean(dim=1)    # 同类平均距离 [B]
        
        # 异类距离计算
        distances_diff = distances[diff_mask].view(distances.size(0), -1)  # [B, M]
        mean_diff_dist = distances_diff.mean(dim=1)     # 异类平均距离 [B]
        
        # 引入间隔约束的损失函数
        loss_nearest = torch.mean(torch.relu(min_same_dist - self.margin))  # 同类最近距离需小于0.5*margin
        loss_intra = torch.mean(mean_same_dist)          # 缩小类内平均距离
        loss_inter = torch.mean(torch.relu(2*self.margin - mean_diff_dist))  # 异类平均距离需大于margin
        
        total_loss = self.nearest_beta*loss_nearest + loss_intra + loss_inter
        return total_loss
    
    
class EmbeddingAnchorLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(EmbeddingAnchorLoss, self).__init__()
        self.margin = margin
        # self.nearest_beta = nearest_beta

    def forward(self, X_anchors, y_anchors):
        
        # 创建一个mask，用于标识标签2与标签3的样本对
        # label2_mask = (y_anchors == 2).unsqueeze(1)
        # label3_mask = (y_anchors == 3).unsqueeze(0)
        # max_mask = label2_mask & label3_mask
        X_anchors = F.normalize(X_anchors, p=2, dim=1)  # [L, D]
        
        distances = torch.cdist(X_anchors.unsqueeze(0), X_anchors.unsqueeze(0)).squeeze()
        
        same_mask = (y_anchors.unsqueeze(0) == y_anchors.unsqueeze(1))  # [B, L]
        diff_mask = ~same_mask
        
        loss_same_label = same_mask.float() * distances
        loss_diff_label = diff_mask.float() * distances
        # loss_2_3 = max_mask.float() * distances
        
        # 总损失，排除对角线元素（即样本与自身的比较）
        # loss = loss_same_label + loss_diff_label
        loss = loss_same_label - loss_diff_label
        loss = loss[~torch.eye(loss.size(0), dtype=bool)].view(loss.size(0), -1)  # 去除对角线元素
        loss = loss.mean()

        return loss
    