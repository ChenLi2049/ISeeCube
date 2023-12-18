import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# regression
def mse_loss(pred, y):
    p_xyz = pred.float()
    
    sa2 = torch.sin(y["target"][:, 0])
    ca2 = torch.cos(y["target"][:, 0])
    sz2 = torch.sin(y["target"][:, 1])
    cz2 = torch.cos(y["target"][:, 1])
    t_xyz = torch.stack([ca2 * sz2, sa2 * sz2, cz2], -1)
    
    loss = nn.MSELoss()(p_xyz, t_xyz)
    return loss

def competition_metric_reg(pred, y):
    pred = F.normalize(pred.double(), dim=-1)

    sa2 = torch.sin(y["target"][:, 0])#sin(azimuth)
    ca2 = torch.cos(y["target"][:, 0])
    sz2 = torch.sin(y["target"][:, 1])#sin(zenith)
    cz2 = torch.cos(y["target"][:, 1])

    scalar_prod = (
        pred[:, 0] * ca2 * sz2 + pred[:, 1] * sa2 * sz2 + pred[:, 2] * cz2
    ).clip(-1 + 1e-8, 1 - 1e-8)
    loss = torch.acos(scalar_prod).abs().mean(-1).float()
    return loss

def loss_comb_reg(pred, y):
    return competition_metric_reg(pred, y) + 0.05 * mse_loss(pred, y)

# classification
def ce_loss(pred, y):
    pred_azi = pred["azimuth"]# [batch_size, 128]
    pred_zen = pred["zenith"]# [batch_size, 64]
    
    bins_width = 2 * math.pi / 128# bins_width = math.pi / 64
    target_azi = y["target"][:, 0]
    bins_index_azi = torch.floor(target_azi / bins_width).long()# [batch_size,]
    target_zen = y["target"][:, 1]
    bins_index_zen = torch.floor(target_zen / bins_width).long()# [batch_size,]
    
    loss = F.cross_entropy(pred_azi, bins_index_azi) + F.cross_entropy(pred_zen, bins_index_zen)
    return loss

def competition_metric_cls(pred, y):
    def classes_to_angle(classes):
        # classes: [batch_size, 128] or [batch_size, 64], angle: [batch_size,]
        classes = F.log_softmax(classes, dim=1)
        max_index = torch.argmax(classes, dim=1)
        angle = (max_index / 128) * 2 * math.pi# angle = (max_index / 64) * math.pi
        return angle
    
    pred_azi = classes_to_angle(pred["azimuth"])
    pred_zen = classes_to_angle(pred["zenith"])
    
    sa1 = torch.sin(pred_azi)#sin(azimuth)
    ca1 = torch.cos(pred_azi)
    sz1 = torch.sin(pred_zen)#sin(zenith)
    cz1 = torch.cos(pred_zen)

    sa2 = torch.sin(y["target"][:, 0])#sin(azimuth)
    ca2 = torch.cos(y["target"][:, 0])
    sz2 = torch.sin(y["target"][:, 1])#sin(zenith)
    cz2 = torch.cos(y["target"][:, 1])

    scalar_prod = (
        ca1 * sz1 * ca2 * sz2 + sa1 * sz1 * sa2 * sz2 + cz1 * cz2
    ).clip(-1 + 1e-8, 1 - 1e-8)
    loss = torch.acos(scalar_prod).abs().mean(-1).float()
    return loss

def loss_comb_cls(pred, y):
    return competition_metric_cls(pred, y) + 0.05 * ce_loss(pred, y)