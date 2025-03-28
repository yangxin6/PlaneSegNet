# -*- coding:utf-8 -*-
"""
coding   : utf-8
@Project ：Pointcept-1.5.1 
@File    ：sp_pt3.py
@IDE     ：PyCharm 
@Author  ：yangxin6
@Date    ：2024/8/3 上午9:06 
"""
from functools import partial
import spconv.pytorch as spconv
import torch
from collections import OrderedDict
from spconv.pytorch.modules import SparseModule
from torch import nn
from typing import Callable, Dict, List, Optional, Union
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max

from pointcept.models import offset2batch
from pointcept.models.builder import MODELS, build_model

@MODELS.register_module("SP2UNet-v1m1")
class SP2UNet(nn.Module):
    def __init__(
            self,
            pre_backbone,
            pre_backbone_out_channels=32,
            pool='mean',
            decoder=None,
    ):
        super().__init__()
        self.pre_backbone = build_model(pre_backbone)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.output_layer = spconv.SparseSequential(norm_fn(pre_backbone_out_channels), nn.ReLU(inplace=True))
        self.pool = pool
        # decoder
        self.decoder = build_model(decoder)


    def forward(self, data_dict):

        sp_feats = self.extract_feat(data_dict)  # [50288, 32]

        # coord0 = data_dict["coord"]
        # grid_coord0 = data_dict["grid_coord"]
        # feat0 = data_dict["feat"]
        # offset0 = data_dict["offset"]

        offset = data_dict["new_offsets"]
        grid_coord = data_dict["new_grid_coord"]

        data_dict["offset"] = offset
        data_dict["grid_coord"] = grid_coord
        data_dict["feat"] = sp_feats

        feat = self.decoder(data_dict)
        seg_logits = feat[data_dict['superpoint']]
        return seg_logits


    def extract_feat(self, data_dict):
        superpoints = data_dict['superpoint']
        grid_coord = data_dict['grid_coord']
        offset = data_dict["offset"]
        feat = self.pre_backbone(data_dict)
        x = self.output_layer(feat)
        device = feat.device

        batch_offsets = torch.cat((torch.tensor([0], device=device), offset))

        new_x = []  # 用于存储每个批次处理后的特征
        new_offset = [0]  # 新的offset从原始offset的第一个元素开始
        new_grid_coord = []
        for i in range(len(batch_offsets) - 1):
            start_idx, end_idx = batch_offsets[i], batch_offsets[i + 1]
            # 提取当前批次的特征和superpoints
            batch_x = x[start_idx:end_idx]
            batch_superpoints = superpoints[start_idx:end_idx]
            batch_grid_coord = grid_coord[start_idx:end_idx]
            # 确保batch_superpoints是从0开始的连续整数，这对scatter_*函数很重要
            # _, batch_superpoints = batch_superpoints.unique(return_inverse=True)

            # 根据superpoints进行scatter_mean或scatter_max操作
            if self.pool == 'mean':
                batch_new_x = scatter_mean(batch_x, batch_superpoints, dim=0, dim_size=batch_superpoints.max() + 1)
                batch_new_grid_coord = scatter_mean(batch_grid_coord, batch_superpoints, dim=0, dim_size=batch_superpoints.max() + 1)
            elif self.pool == 'max':
                batch_new_x, _ = scatter_max(batch_x, batch_superpoints, dim=0, dim_size=batch_superpoints.max() + 1)
                batch_new_grid_coord = scatter_max(batch_grid_coord, batch_superpoints, dim=0, dim_size=batch_superpoints.max() + 1)

            new_x.append(batch_new_x)
            new_grid_coord.append(batch_new_grid_coord)

            new_offset.append(new_offset[-1] + batch_new_x.size(0))

        new_offset = new_offset[1:]
        new_x = torch.cat(new_x, dim=0).to(device)
        new_grid_coord = torch.cat(new_grid_coord, dim=0).to(device)
        new_offset = torch.tensor(new_offset, device=device)  # 指定设备
        data_dict["new_offsets"] = new_offset
        data_dict["new_grid_coord"] = new_grid_coord
        return new_x

