#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 18:22
# @Author  : yangxin
# @Email   : yangxinnc@163.com
# @File    : corn3d_ins.py
# @Project : PlantPointSeg
# @Software: PyCharm
import glob
import os
from copy import deepcopy

import numpy as np
import torch
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Huasheng3dDataset(DefaultDataset):
    class2id = np.array([0, 1, 2])

    def __init__(
            self,
            split="train",
            data_root="data/huasheng3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data(self, idx):
        if self.split == "predict":
            data_path = self.data_list[idx % len(self.data_list)]
            data = np.loadtxt(data_path)
            coord = data[:, :3]
            normal = data[:, 3:6]
            color = np.zeros_like(coord)
            segment = np.ones(coord.shape[0]) * -1
            instance = np.ones(coord.shape[0]) * -1
            scene_id = os.path.basename(data_path)[:-4]
            data_dict = dict(
                coord=coord,
                normal=normal,
                color=color,
                segment=segment,
                instance=instance,
                scene_id=scene_id,
            )
            return data_dict
        else:
            data_path = self.data_list[idx % len(self.data_list)]
            data = torch.load(data_path)
            coord = data["coord"]
            color = data["color"]
            normal = data["normal"]
            scene_id = data["scene_id"]
            if "semantic_gt" in data.keys():
                segment = data["semantic_gt"].reshape([-1]).astype(int)
            else:
                segment = np.ones(coord.shape[0]) * -1
            if "instance_gt" in data.keys():
                instance = data["instance_gt"].reshape([-1]).astype(int)
            else:
                instance = np.ones(coord.shape[0]) * -1
            data_dict = dict(
                coord=coord,
                normal=normal,
                color=color,
                segment=segment,
                instance=instance,
                scene_id=scene_id,
            )
            if "superpoint" in data.keys():
                superpoint = data["superpoint"]
                data_dict["superpoint"] = superpoint
            if "pred" in data.keys():
                pred = data["pred"]
                data_dict["pred"] = pred
            return data_dict

