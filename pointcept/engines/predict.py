"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
from uuid import uuid4

import numpy as np
from collections import OrderedDict

import pointops
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from pointgroup_ops import ballquery_batch_p, bfs_cluster
from pointcept.models.utils import offset2batch, batch2offset
from torch import nn

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

PREDICTS = Registry("predicts")


class PredictBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg

        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@PREDICTS.register_module()
class SemSegTester(PredictBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result/sem_pred_label")
        make_dirs(save_path)
        # create submit folder only on main process

        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            data_name = data_dict.pop("name")
            coord = data_dict['coord']
            pred_save_path = os.path.join(save_path, "{}_pred.txt".format(data_name))

            pred = torch.zeros((coord.shape[0], self.cfg.data.num_classes)).cuda()
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad():
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        batch_idx=i,
                        batch_num=len(fragment_list),
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            np.savetxt(pred_save_path, pred, fmt="%d")

        logger.info("Test Finished!")

    @staticmethod
    def collate_fn(batch):
        return batch




@PREDICTS.register_module()
class InstanceSegTester2(PredictBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.segment_ignore_index = self.model.segment_ignore_index
        self.instance_ignore_index = self.model.instance_ignore_index
        self.instance_segment_ignore_index = self.model.instance_segment_ignore_index

        self.valid_class_names = [
            self.cfg.data.names[i]
            for i in range(self.cfg.data.num_classes)
            if i not in self.instance_segment_ignore_index
        ]
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

        self.voxel_size = self.model.voxel_size
        self.cluster_thresh = self.model.cluster_thresh
        self.cluster_closed_points = self.model.cluster_closed_points
        self.cluster_min_points = self.model.cluster_min_points
        self.cluster_propose_points = self.model.cluster_propose_points

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Prediction >>>>>>>>>>>>>>>>")

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result", "inst_pred_label")
        make_dirs(save_path)


        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]
            fragment_list = data_dict.pop("fragment_list")
            # segment = data_dict.pop("segment")
            # instance = data_dict.pop("instance")
            data_name = data_dict['name']
            coord = data_dict['coord']

            fragment_infer_time = 0
            # for i in range(len(fragment_list)):
            fragment_batch_size = 1
            i = 0
            s_i, e_i = i * fragment_batch_size, min(
                (i + 1) * fragment_batch_size, len(fragment_list)
            )
            input_dict = collate_fn(fragment_list[s_i:e_i])
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            # idx_part = input_dict["index"]
            with torch.no_grad():
                part_output_dict = self.model(input_dict)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                part_output_dict['pred_masks'] = part_output_dict['pred_masks'][:, input_dict["inverse"].cpu()]
                part_output_dict['bias_pred'] = part_output_dict['bias_pred'][input_dict["inverse"].cpu()]

            idx_model_infer_time = time.time() - end
            fragment_infer_time += idx_model_infer_time
            pred_inst = np.ones((coord.shape[0], 2)) * -1
            for instance_id, mask in enumerate(part_output_dict['pred_masks']):
                semantic_class = part_output_dict['pred_classes'][instance_id].item()
                pred_inst[mask == 1, 0] = semantic_class
                pred_inst[mask == 1, 1] = instance_id  # 实例ID从0开始

            pred_save_path = os.path.join(save_path, f"{data_name}_pred.txt")
            np.savetxt(pred_save_path, pred_inst, fmt="%d")

            logger.info(f"save path: {pred_save_path}")
        logger.info("<<<<<<<<<<<<<<<<< End Prediction <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
