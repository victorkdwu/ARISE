import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class autoencoder(nn.Module):
    def __init__(self, input_size=105, hidden_size=32):
        super(autoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # self.flatten = nn.Sequential(
        #     nn.Flatten()
        # )
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size*2),
            nn.BatchNorm1d(self.hidden_size*2),
            hswish(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            hswish()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.BatchNorm1d(self.hidden_size*2),
            hswish(),
            nn.Linear(self.hidden_size*2, self.input_size),
            nn.BatchNorm1d(self.input_size),
            hswish()
        )
        
    def forward(self, x):
        # x = self.flatten(x)
        # x = torch.unsqueeze(x, dim=1)
        hidden = self.encoder(x)
        rect = self.decoder(hidden).squeeze()
        return rect
    
    def encode(self, x):
        # x = self.flatten(x)
        # x = torch.unsqueeze(x, dim=1)
        hidden = self.encoder(x)
        rect = self.decoder(hidden).squeeze()
        return hidden.squeeze(), rect
    
    def excute_RMSE(self, input, target):
        mse = (input - target).pow(2).sum(1) / self.input_size
        rmse = torch.sqrt(mse)
        return rmse.detach().cpu().numpy().reshape(-1)  # 0 :error 1:normal

class EmbeddingAE(nn.Module):
    def __init__(self, input_size=105, hidden_size=32, pred_k=7):
        super(EmbeddingAE, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pred_k = pred_k
        
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
        hidden = self.encoder(x).squeeze(dim=1)
        return hidden
    
    def pred(self, inputs, X_anchors, y_anchors):
        hidden_inputs = self.forward(inputs)
        hidden_anchors = self.forward(X_anchors)
        
        distances = torch.cdist(hidden_inputs, hidden_anchors)
        topk_indices = torch.topk(distances, k=self.pred_k, dim=1, largest=False).indices
        
        votes = y_anchors[topk_indices].cpu().numpy()
        label = [Counter(row).most_common(1)[0][0] for row in votes]
        
        return label, topk_indices.squeeze().cpu().numpy().tolist()


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

    def forward(self, X_inputs, y_inputs, X_anchors, y_anhcors):
        distances = torch.cdist(X_inputs.unsqueeze(0), X_anchors.unsqueeze(0)).squeeze()
        distances_same_label = distances[torch.nonzero(y_inputs == y_anhcors)].squeeze(1)
        distances_diff_label = distances[torch.nonzero(y_inputs != y_anhcors)].squeeze(1)
        
        loss_embedding = self.nearest_beta*torch.mean(torch.clamp(distances_same_label.min(dim=1)[0] - self.margin, min=0.0))\
            + torch.mean(distances_same_label.mean(dim=1))\
            - torch.mean(torch.clamp(distances_diff_label.mean(dim=1) - self.margin * 2, max=0.0))
            
        # loss_embedding = torch.mean(torch.clamp(distances_same_label.min(dim=1)[0] - self.margin, min=0.0))\
        #     - torch.mean(torch.clamp(distances_diff_label.mean(dim=1) - self.margin * 2, max=0.0))
        
        # loss_embedding = torch.mean(distances_same_label.mean(dim=1))\
        #     - torch.mean(torch.clamp(distances_diff_label.mean(dim=1) - self.margin * 2, max=0.0))
        return loss_embedding
    