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

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors

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
from ..utils.post_inference_v2 import post_process

TESTERS = Registry("testers")


class TesterBase:
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


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result/sem_pred")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        f_csv = open(os.path.join(save_path, "sem_res_total.csv"), "w")
        f_csv.write("Data Name,mIoU,Acc,mAcc,times,")
        for i in range(self.cfg.data.num_classes):
            f_csv.write("clas Name,IoU,Acc,")
        f_csv.write("\n")
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            coord = data_dict['coord']
            pred_save_path = os.path.join(save_path, "{}_pred.txt".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred_data = np.loadtxt(pred_save_path)
                pred = pred_data[:, 3]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
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
                pred_data = np.ones((coord.shape[0], 5)) * -1
                pred_data[:, :3] = coord
                pred_data[:, 3] = pred
                pred_data[:, 4] = segment
                np.savetxt(pred_save_path, pred_data)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            f_csv.write(f"{data_name},{iou},{acc},{m_acc},{batch_time.avg},")
            for i in range(self.cfg.data.num_classes):
                f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            f_csv.write("\n")
            if (
                    self.cfg.data.test.type == "ScanNetDataset"
                    or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            # f_csv.write(f"AVG,{mIoU},{allAcc},{mAcc},")
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
                # f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class SemSegTesterFile(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result/sem_pred")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        f_csv = open(os.path.join(save_path, "sem_res_total.csv"), "w")
        f_csv.write("Data Name,mIoU,Acc,mAcc,")
        for i in range(self.cfg.data.num_classes):
            f_csv.write("clas Name,IoU,Acc,")
        f_csv.write("\n")
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            pred_label = data_dict.pop("pred")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            coord = data_dict['coord']
            pred_save_path = os.path.join(save_path, "{}_pred.txt".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred_data = np.loadtxt(pred_save_path)
                pred = pred_data[:, 3]
            else:
                # pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                pred = pred_label

                pred_data = np.ones((coord.shape[0], 5)) * -1
                pred_data[:, :3] = coord
                pred_data[:, 3] = pred
                pred_data[:, 4] = segment
                np.savetxt(pred_save_path, pred_data)

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            f_csv.write(f"{data_name},{iou},{acc},{m_acc},")
            for i in range(self.cfg.data.num_classes):
                f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            f_csv.write("\n")
            if (
                    self.cfg.data.test.type == "ScanNetDataset"
                    or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            # f_csv.write(f"AVG,{mIoU},{allAcc},{mAcc},")
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
                # f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch



@TESTERS.register_module()
class SemSegInstPostTester(TesterBase):
    def test(self):
        save_inst = True
        save_inst_path = None
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result/sem_pred")
        make_dirs(save_path)
        if save_inst:
            save_inst_path = os.path.join(self.cfg.save_path, "result/sem_pred/inst_pred")
            make_dirs(save_inst_path)

        # create submit folder only on main process
        if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        f_csv = open(os.path.join(save_path, "sem_res_total.csv"), "w")
        f_csv.write("Data Name,mIoU,Acc,mAcc,")
        for i in range(self.cfg.data.num_classes):
            f_csv.write("clas Name,IoU,Acc,")
        f_csv.write("\n")
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            coord = data_dict['coord']
            pred_save_path = os.path.join(save_path, "{}_pred.txt".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred_data = np.loadtxt(pred_save_path)
                pred = pred_data[:, 3]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
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
                        pred_dict = self.model(input_dict)
                        pred_part = self.post_seg(input_dict, pred_dict, fra_i=i, save_inst=save_inst_path)
                        # pred_part = self.post_seg_TC(input_dict, pred_dict, fra_i=i, save_inst=save_inst_path)

                        # pred_part = pred_dict["seg_logits"]  # (n, k)
                        # pred_part = F.softmax(pred_part, -1)
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
                pred_data = np.ones((coord.shape[0], 5)) * -1
                pred_data[:, :3] = coord
                pred_data[:, 3] = pred
                pred_data[:, 4] = segment
                np.savetxt(pred_save_path, pred_data)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            f_csv.write(f"{data_name},{iou},{acc},{m_acc},")
            for i in range(self.cfg.data.num_classes):
                f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            f_csv.write("\n")
            if (
                    self.cfg.data.test.type == "ScanNetDataset"
                    or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            # f_csv.write(f"AVG,{mIoU},{allAcc},{mAcc},")
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
                # f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

            # 生成曲面

    def compute_z_median_stats(self, coord, pred_inst0, e=0.04):
        unique_instances = np.unique(pred_inst0[pred_inst0 >= 1])  # 获取所有实例ID，忽略非实例值 (-1, 0, -2, -3等)

        z_min_values = []
        z_mean_values = []

        for instance_id in unique_instances:
            mask = pred_inst0 == instance_id  # 获取该实例的掩码
            instance_coords = coord[mask]  # 提取该实例的坐标
            z_values = instance_coords[:, 2]  # 提取Z轴的值

            if len(z_values) > 0:
                z_min_values.append(np.min(z_values))  # 记录该实例的Z轴最小值
                z_mean_values.append(np.mean(z_values))  # 记录该实例的Z轴平均值

        # 计算所有实例Z轴最小值和平均值的中位数
        z_min_median = np.median(z_min_values) if z_min_values else None
        z_mean_median = np.median(z_mean_values) if z_mean_values else None

        return z_min_median+e, z_mean_median

    def post_seg(self, input_dict, pred_dict, z_th=0.5, fra_i=0, save_inst=None):
        data_name = input_dict["name"][0]
        coord = input_dict["coord"].cpu().numpy()
        z_th = min(coord[:, 2].max()-0.05, z_th)
        pred_masks = pred_dict["pred_masks"].cpu().numpy()
        pred_classes = pred_dict["pred_classes"].cpu().numpy()

        pred_logits = pred_dict["seg_logits"]
        pred_sem_class = F.softmax(pred_logits, -1)
        pred_sem_class = pred_sem_class.max(1)[1].data.cpu().numpy()
        ground_mask = pred_sem_class == 0

        pred_inst0 = np.ones_like(pred_sem_class) * -1
        pred_inst0[ground_mask] = 0
        for instance_id, mask in enumerate(pred_masks):
            semantic_class = pred_classes[instance_id]
            mask_coord = coord[mask==1]
            # mask_coord = self.sor_filter(mask_coord, k=16, z_threshold=1.0)  # k 和 z_threshold 可以根据情况调整
            if mask_coord[:, 2].max() < z_th:
                semantic_class = -2
                instance_id = -3
            pred_inst0[mask == 1] = instance_id + 1  # 实例ID从1开始

        z_min_median, z_mean_median = self.compute_z_median_stats(coord, pred_inst0)

        pred = np.ones_like(pred_sem_class) * -1
        pred_inst = np.ones_like(pred_sem_class) * -1
        pred[ground_mask] = 0
        pred_inst[ground_mask] = 0
        # 根据 surface_z_mean 标记地面点
        for instance_id, mask in enumerate(pred_masks):
            mask_coord = coord[mask == 1]
            semantic_class = pred_classes[instance_id]

            if mask_coord[:, 2].max() < z_mean_median:
                semantic_class = -2
                instance_id = -3
            pred_inst[mask == 1] = instance_id + 1  # 实例ID从1开始
            pred[mask == 1] = semantic_class

        pred_zero_mask = pred == 0
        coord_zero = coord[pred_zero_mask]
        # show_pcd(coord[pred_zero_mask], pred[pred_zero_mask])

        if len(coord_zero) > 0:
            mask_above_surface_z_min = coord_zero[:, 2] > z_min_median

            pred[pred_zero_mask] = np.where(mask_above_surface_z_min, -1, pred[pred_zero_mask])
            pred_inst[pred_zero_mask] = np.where(mask_above_surface_z_min, -1, pred_inst[pred_zero_mask])

        # show_pcd(coord, pred)
        # show_pcd(coord, pred_inst)
        # 将 pred_inst == -1 的点根据 z_min_median 判断为地面点
        inst_neg1_mask = pred_inst == -1
        coord_neg1 = coord[inst_neg1_mask]
        if len(coord_neg1) > 0:
            # 对每个 pred_inst == -1 的点，利用其 X 和 Y 坐标通过 griddata 查找相应位置的 surface_z_min 值

            # 找出 Z 值小于 z_mean_median 的点，将其视为地面点
            ground_mask_neg1 = coord_neg1[:, 2] > z_mean_median

            # 更新 pred 和 pred_inst 中的地面标记
            # 直接在原数组上进行修改
            pred[inst_neg1_mask] = np.where(ground_mask_neg1, 1, pred[inst_neg1_mask])
            pred_inst[inst_neg1_mask] = np.where(ground_mask_neg1, 1, pred_inst[inst_neg1_mask])

        # show_pcd(coord, pred)
        # show_pcd(coord, pred_inst)



        # show_pcd(coord, pred)
        # show_pcd(coord, pred_inst)

        post_ground_mask = pred <= 0
        pred_logits[post_ground_mask, 0] = pred_logits.max()  # 设置地面类别的 logits 为高值
        pred_logits[post_ground_mask, 1] = pred_logits.min()  # 非地面类别的 logits 设置为低值

        if fra_i == 0 and save_inst is not None:
            np.savetxt(os.path.join(save_inst, f"{data_name}_{fra_i}_pred.txt"), np.c_[coord, pred_inst])
        return pred_logits

    def post_seg_TC(self, input_dict, pred_dict, z_th=0.1, e=0.02, fra_i=0, save_inst=None):
        data_name = input_dict["name"][0]
        coord = input_dict["coord"].cpu().numpy()
        pred_masks = pred_dict["pred_masks"].cpu().numpy()
        pred_classes = pred_dict["pred_classes"].cpu().numpy()

        pred_logits = pred_dict["seg_logits"]
        pred_sem_class = F.softmax(pred_logits, -1)
        pred_sem_class = pred_sem_class.max(1)[1].data.cpu().numpy()
        ground_mask = pred_sem_class == 0

        pred_inst0 = np.ones_like(pred_sem_class) * -1
        pred_inst0[ground_mask] = 0
        for instance_id, mask in enumerate(pred_masks):
            semantic_class = pred_classes[instance_id]
            mask_coord = coord[mask==1]
            # mask_coord = self.sor_filter(mask_coord, k=16, z_threshold=1.0)  # k 和 z_threshold 可以根据情况调整
            if mask_coord[:, 2].max() < z_th:
                semantic_class = -2
                instance_id = -3
            pred_inst0[mask == 1] = instance_id + 1  # 实例ID从1开始

        z_min_median, z_mean_median = self.compute_z_median_stats(coord, pred_inst0, e=e)

        pred = np.ones_like(pred_sem_class) * -1
        pred_inst = np.ones_like(pred_sem_class) * -1
        pred[ground_mask] = 0
        pred_inst[ground_mask] = 0
        # 根据 surface_z_mean 标记地面点
        for instance_id, mask in enumerate(pred_masks):
            mask_coord = coord[mask == 1]
            semantic_class = pred_classes[instance_id]

            # if mask_coord[:, 2].max() < z_mean_median:
            #     semantic_class = -2
            #     instance_id = -3
            pred_inst[mask == 1] = instance_id + 1  # 实例ID从1开始
            pred[mask == 1] = semantic_class

        pred_zero_mask = pred == 0
        coord_zero = coord[pred_zero_mask]
        # show_pcd(coord[pred_zero_mask], pred[pred_zero_mask])

        if len(coord_zero) > 0:
            mask_above_surface_z_min = coord_zero[:, 2] > z_min_median

            pred[pred_zero_mask] = np.where(mask_above_surface_z_min, -1, pred[pred_zero_mask])
            pred_inst[pred_zero_mask] = np.where(mask_above_surface_z_min, -1, pred_inst[pred_zero_mask])

        # show_pcd(coord, pred)
        # show_pcd(coord, pred_inst)
        # 将 pred_inst == -1 的点根据 z_min_median 判断为地面点
        inst_neg1_mask = pred_inst == -1
        coord_neg1 = coord[inst_neg1_mask]
        if len(coord_neg1) > 0:
            # 对每个 pred_inst == -1 的点，利用其 X 和 Y 坐标通过 griddata 查找相应位置的 surface_z_min 值

            # 找出 Z 值小于 z_mean_median 的点，将其视为地面点
            ground_mask_neg1 = coord_neg1[:, 2] > z_mean_median

            # 更新 pred 和 pred_inst 中的地面标记
            # 直接在原数组上进行修改
            pred[inst_neg1_mask] = np.where(ground_mask_neg1, 1, pred[inst_neg1_mask])
            pred_inst[inst_neg1_mask] = np.where(ground_mask_neg1, 1, pred_inst[inst_neg1_mask])

        # show_pcd(coord, pred)
        # show_pcd(coord, pred_inst)



        # show_pcd(coord, pred)
        # show_pcd(coord, pred_inst)

        post_ground_mask = pred <= 0
        pred_logits = torch.zeros_like(pred_logits)
        pred_logits[post_ground_mask, 0] = 1
        pred_logits[~post_ground_mask, 1] = 1
        # show_pcd(coord[post_ground_mask], pred[post_ground_mask])
        # pred_logits[post_ground_mask, 0] = 1  # 设置地面类别的 logits 为高值
        # pred_logits[post_ground_mask, 1] = pred_logits.min()  # 非地面类别的 logits 设置为低值

        if fra_i == 0 and save_inst is not None:
            np.savetxt(os.path.join(save_inst, f"{data_name}_{fra_i}_pred.txt"), np.c_[coord, pred_inst])
        return pred_logits

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class SemSegTester2(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result/sem_pred")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        f_csv = open(os.path.join(save_path, "sem_res_total.csv"), "w")
        f_csv.write("Data Name,mIoU,Acc,mAcc,")
        for i in range(self.cfg.data.num_classes):
            f_csv.write("clas Name,IoU,Acc,")
        f_csv.write("\n")
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            coord = data_dict['coord']
            pred_save_path = os.path.join(save_path, "{}_pred.txt".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred_data = np.loadtxt(pred_save_path)
                pred = pred_data[:, 3]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
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

                pred_data = np.ones((coord.shape[0], 5)) * -1
                pred_data[:, :3] = coord
                pred_data[:, 3] = pred
                pred_data[:, 4] = segment
                np.savetxt(pred_save_path, pred_data)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            f_csv.write(f"{data_name},{iou},{acc},{m_acc},")
            for i in range(self.cfg.data.num_classes):
                f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            f_csv.write("\n")
            if (
                    self.cfg.data.test.type == "ScanNetDataset"
                    or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            # f_csv.write(f"AVG,{mIoU},{allAcc},{mAcc},")
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
                # f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class SemSegTester3(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result/sem_pred")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        f_csv = open(os.path.join(save_path, "sem_res_total.csv"), "w")
        f_csv.write("Data Name,mIoU,Acc,mAcc,")
        for i in range(self.cfg.data.num_classes):
            f_csv.write("clas Name,IoU,Acc,")
        f_csv.write("\n")
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            coord = data_dict['coord']
            pred_save_path = os.path.join(save_path, "{}_pred.txt".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred_data = np.loadtxt(pred_save_path)
                pred = pred_data[:, 3]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes+1)).cuda()
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
                pred[pred == self.cfg.data.num_classes] = -1

                pred_data = np.ones((coord.shape[0], 5)) * -1
                pred_data[:, :3] = coord
                pred_data[:, 3] = pred
                pred_data[:, 4] = segment
                np.savetxt(pred_save_path, pred_data)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            f_csv.write(f"{data_name},{iou},{acc},{m_acc},")
            for i in range(self.cfg.data.num_classes):
                f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            f_csv.write("\n")
            if (
                    self.cfg.data.test.type == "ScanNetDataset"
                    or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            # f_csv.write(f"AVG,{mIoU},{allAcc},{mAcc},")
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
                # f_csv.write(f"{self.cfg.data.names[i]},{iou_class[i]},{accuracy_class[i]},")
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class InstanceSegTester(TesterBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.segment_ignore_index = self.model.segment_ignore_index
        self.instance_ignore_index = self.model.instance_ignore_index

        self.valid_class_names = [
            self.cfg.data.names[i]
            for i in range(self.cfg.data.num_classes)
            if i not in self.segment_ignore_index
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

    def associate_instances(self, pred, segment, instance):
        # segment = segment.cpu().numpy().astype(int)
        # instance = instance.cpu().numpy().astype(int)
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
                pred["pred_classes"].shape[0]
                == pred["pred_scores"].shape[0]
                == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
                zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                               and gt["med_dist"] <= distance_thresh
                               and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                        gt["vert_count"]
                                        + pred["vert_count"]
                                        - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                        gt["vert_count"]
                                        + pred["vert_count"]
                                        - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                            gt["vert_count"] < min_region_size
                                            or gt["med_dist"] > distance_thresh
                                            or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                        float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def evaluate_bias(self, bias_pred, coord, instance):
        center_pred = bias_pred + coord
        center_radius = np.zeros(len(center_pred))
        unique_inst = np.unique(instance)
        out_dist_item = []
        out_num = 0
        for i in unique_inst:
            i_mask = i == instance
            i_coord = coord[i_mask]
            i_center_pred = center_pred[i_mask]
            i_instance_centroid = i_coord.mean(0)
            dist = np.linalg.norm(i_center_pred[:, :2] - i_instance_centroid[:2], axis=1)
            center_radius[i_mask] = dist
        return center_pred, center_radius

    def test(self):
        infer_once = True
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result", "inst_pred")
        make_dirs(save_path)

        f_csv = open(os.path.join(save_path, "inst_res_total.csv"), "w")
        f_csv.write("Data Name,AP,AP50\n")

        scenes = []
        total_infer_time = 0
        all_data_avg_ap = []
        all_data_std_ap = []
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            instance = data_dict.pop("instance")
            data_name = data_dict['name']
            coord = data_dict['coord']

            # only inference
            # infer = (instance < 0).all()

            best_output = None
            best_one_ap_scores = None
            best_one_scene = None
            best_ap = -1
            total_ap = []
            # best_ap50 = -1
            # total_ap50 = []
            fragment_infer_time = 0
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
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

                    # instance proposal
                    # output_dict = self.pred_feat(pred_segment, pred_bias, coord)

                idx_model_infer_time = time.time() - end
                fragment_infer_time += idx_model_infer_time

                gt_instances, pred_instance = self.associate_instances(
                    part_output_dict, segment, instance
                )
                one_scene = dict(gt=gt_instances, pred=pred_instance)
                # scenes.append(one_scene)

                one_ap_scores = self.evaluate_matches([one_scene])
                one_all_ap = one_ap_scores["all_ap"]
                one_all_ap_50 = one_ap_scores["all_ap_50%"]
                one_all_ap_25 = one_ap_scores["all_ap_25%"]
                # total_ap50.append(one_all_ap_50)
                total_ap.append(one_all_ap)
                if one_all_ap > best_ap or infer_once:
                    best_output = part_output_dict
                    best_ap = one_all_ap
                    best_one_ap_scores = one_ap_scores
                    best_one_scene = one_scene

                logger.info(
                    "Test: {}/{}-{}, item: {}/{} mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}; Point Size: {} Batch {:.3f}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name,
                        i + 1,
                        len(fragment_list),
                        one_all_ap,
                        one_all_ap_50,
                        one_all_ap_25,
                        segment.shape[0],
                        idx_model_infer_time,
                    )
                )
                if infer_once:
                    break
            scenes.append(best_one_scene)
            best_ap50 = best_one_ap_scores['all_ap_50%']
            f_csv.write(f"{data_name},{best_ap},{best_ap50}\n")

            # 对 'bias_pred' 进行评估
            pred_center, center_radius = self.evaluate_bias(best_output['bias_pred'].cpu().numpy(), coord, instance)

            # 保存每个分割的结果
            pred_data = np.ones((coord.shape[0], 10)) * -1
            pred_data[:, :3] = coord
            pred_data[:, 3:6] = pred_center
            pred_data[:, 6] = center_radius
            pred_data[:, 9] = instance
            for instance_id, mask in enumerate(best_output['pred_masks']):
                semantic_class = best_output['pred_classes'][instance_id].item()
                pred_data[mask == 1, 7] = semantic_class
                pred_data[mask == 1, 8] = instance_id  # 实例ID从0开始
            pred_save_path = os.path.join(save_path, f"{data_name}_pred.txt")
            np.savetxt(pred_save_path, pred_data)
            # if infer:
            #     logger.info(
            #         "Test: {} [{}/{}] Point Size: {} Batch {:.3f} Save Path: {}".format(
            #             data_name,
            #             idx + 1,
            #             len(self.test_loader),
            #             coord.shape[0],
            #             fragment_infer_time / len(fragment_list),
            #             pred_save_path,
            #         )
            #     )
            #     continue
            avg_ap = np.mean(total_ap)
            std_ap = np.std(total_ap)
            all_data_avg_ap.append(avg_ap)
            all_data_std_ap.append(std_ap)
            logger.info(
                "Test: {} [{}/{}] Result mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}; avg AP50 {:.4f} std:{:.4f} Point Size: {} Batch {:.3f} Save Path: {}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    best_one_ap_scores["all_ap"],
                    best_one_ap_scores["all_ap_50%"],
                    best_one_ap_scores["all_ap_25%"],
                    avg_ap,
                    std_ap,
                    segment.shape[0],
                    fragment_infer_time / len(fragment_list),
                    pred_save_path,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        all_avg_ap = np.mean(all_data_avg_ap)
        all_std_ap = np.mean(all_data_std_ap)
        logger.info(
            "Test result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f} avg mAP {:.4f} std:{:.4f} batch_time avg: {:.3f}.".format(
                all_ap, all_ap_50, all_ap_25, all_avg_ap, all_std_ap, total_infer_time / len(self.test_loader)
            )
        )

        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch



@TESTERS.register_module()
class InstanceSegTester2(TesterBase):

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

    def associate_instances(self, pred, segment, instance):
        # segment = segment.cpu().numpy().astype(int)
        # instance = instance.cpu().numpy().astype(int)
        void_mask = np.in1d(segment, self.instance_segment_ignore_index)

        assert (
                pred["pred_classes"].shape[0]
                == pred["pred_scores"].shape[0]
                == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.instance_segment_ignore_index:
                gt_instances[self.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.instance_segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.instance_segment_ignore_index:
                pred_instances[self.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.instance_segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
                zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                               and gt["med_dist"] <= distance_thresh
                               and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                        gt["vert_count"]
                                        + pred["vert_count"]
                                        - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                        gt["vert_count"]
                                        + pred["vert_count"]
                                        - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.instance_segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                            gt["vert_count"] < min_region_size
                                            or gt["med_dist"] > distance_thresh
                                            or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                        float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def evaluate_bias(self, bias_pred, coord, instance):
        center_pred = bias_pred + coord
        center_radius = np.zeros(len(center_pred))
        unique_inst = np.unique(instance)
        out_dist_item = []
        out_num = 0
        for i in unique_inst:
            i_mask = i == instance
            i_coord = coord[i_mask]
            i_center_pred = center_pred[i_mask]
            i_instance_centroid = i_coord.mean(0)
            dist = np.linalg.norm(i_center_pred[:, :2] - i_instance_centroid[:2], axis=1)
            center_radius[i_mask] = dist
        return center_pred, center_radius

    def test(self):
        infer_once = False
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result", "inst_pred")
        make_dirs(save_path)

        f_csv = open(os.path.join(save_path, "inst_res_total.csv"), "w")
        # f_csv.write("Data Name,AP,AP50\n")

        scenes = []
        total_infer_time = 0
        all_data_avg_ap = []
        all_data_std_ap = []
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            instance = data_dict.pop("instance")
            data_name = data_dict['name']
            coord = data_dict['coord']

            # only inference
            # infer = (instance < 0).all()

            best_output = None
            best_one_ap_scores = None
            best_one_scene = None
            best_ap = -1
            total_ap = []
            # best_ap50 = -1
            # total_ap50 = []
            fragment_infer_time = 0
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
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

                    # instance proposal
                    # output_dict = self.pred_feat(pred_segment, pred_bias, coord)

                idx_model_infer_time = time.time() - end
                fragment_infer_time += idx_model_infer_time

                gt_instances, pred_instance = self.associate_instances(
                    part_output_dict, segment, instance
                )
                one_scene = dict(gt=gt_instances, pred=pred_instance)
                # scenes.append(one_scene)

                one_ap_scores = self.evaluate_matches([one_scene])
                one_all_ap = one_ap_scores["all_ap"]
                one_all_ap_50 = one_ap_scores["all_ap_50%"]
                one_all_ap_25 = one_ap_scores["all_ap_25%"]
                # total_ap50.append(one_all_ap_50)
                total_ap.append(one_all_ap)
                if one_all_ap > best_ap or infer_once:
                    best_output = part_output_dict
                    best_ap = one_all_ap
                    best_one_ap_scores = one_ap_scores
                    best_one_scene = one_scene

                logger.info(
                    "Test: {}/{}-{}, item: {}/{} mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}; Point Size: {} Batch {:.3f}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name,
                        i + 1,
                        len(fragment_list),
                        one_all_ap,
                        one_all_ap_50,
                        one_all_ap_25,
                        segment.shape[0],
                        idx_model_infer_time,
                    )
                )
                if infer_once:
                    break
            scenes.append(best_one_scene)
            best_ap50 = best_one_ap_scores['all_ap_50%']
            f_csv.write(f"{data_name},{best_ap},{best_ap50}")
            for nn in best_one_ap_scores['classes']:
                f_csv.write(f",{best_one_ap_scores['classes'][nn]['ap']},{best_one_ap_scores['classes'][nn]['ap50%']}")
            f_csv.write("\n")
            # 对 'bias_pred' 进行评估
            pred_center, center_radius = self.evaluate_bias(best_output['bias_pred'].cpu().numpy(), coord, instance)

            # 保存每个分割的结果
            pred_data = np.ones((coord.shape[0], 11)) * -1
            pred_data[:, :3] = coord
            pred_data[:, 3:6] = pred_center
            pred_data[:, 6] = center_radius
            pred_data[:, 9] = segment
            pred_data[:, 10] = instance
            for instance_id, mask in enumerate(best_output['pred_masks']):
                semantic_class = best_output['pred_classes'][instance_id].item()
                pred_data[mask == 1, 7] = semantic_class
                pred_data[mask == 1, 8] = instance_id  # 实例ID从0开始
            pred_save_path = os.path.join(save_path, f"{data_name}_pred.txt")
            np.savetxt(pred_save_path, pred_data)
            # if infer:
            #     logger.info(
            #         "Test: {} [{}/{}] Point Size: {} Batch {:.3f} Save Path: {}".format(
            #             data_name,
            #             idx + 1,
            #             len(self.test_loader),
            #             coord.shape[0],
            #             fragment_infer_time / len(fragment_list),
            #             pred_save_path,
            #         )
            #     )
            #     continue
            avg_ap = np.mean(total_ap)
            std_ap = np.std(total_ap)
            all_data_avg_ap.append(avg_ap)
            all_data_std_ap.append(std_ap)
            logger.info(
                "Test: {} [{}/{}] Result mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}; avg AP50 {:.4f} std:{:.4f} Point Size: {} Batch {:.3f} Save Path: {}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    best_one_ap_scores["all_ap"],
                    best_one_ap_scores["all_ap_50%"],
                    best_one_ap_scores["all_ap_25%"],
                    avg_ap,
                    std_ap,
                    segment.shape[0],
                    fragment_infer_time / len(fragment_list),
                    pred_save_path,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        all_avg_ap = np.mean(all_data_avg_ap)
        all_std_ap = np.mean(all_data_std_ap)
        logger.info(
            "Test result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f} avg mAP {:.4f} std:{:.4f} batch_time avg: {:.3f}.".format(
                all_ap, all_ap_50, all_ap_25, all_avg_ap, all_std_ap, total_infer_time / len(self.test_loader)
            )
        )

        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch




@TESTERS.register_module()
class InstanceSegTesterDFSP(TesterBase):

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

    def associate_instances(self, pred, segment, instance):
        # segment = segment.cpu().numpy().astype(int)
        # instance = instance.cpu().numpy().astype(int)
        void_mask = np.in1d(segment, self.instance_segment_ignore_index)

        assert (
                pred["pred_classes"].shape[0]
                == pred["pred_scores"].shape[0]
                == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.instance_segment_ignore_index:
                gt_instances[self.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.instance_segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.instance_segment_ignore_index:
                pred_instances[self.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.instance_segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
                zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                               and gt["med_dist"] <= distance_thresh
                               and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                        gt["vert_count"]
                                        + pred["vert_count"]
                                        - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                        gt["vert_count"]
                                        + pred["vert_count"]
                                        - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.instance_segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                            gt["vert_count"] < min_region_size
                                            or gt["med_dist"] > distance_thresh
                                            or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                        float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def evaluate_bias(self, bias_pred, coord, instance):
        center_pred = bias_pred + coord
        center_radius = np.zeros(len(center_pred))
        unique_inst = np.unique(instance)
        out_dist_item = []
        out_num = 0
        for i in unique_inst:
            i_mask = i == instance
            i_coord = coord[i_mask]
            i_center_pred = center_pred[i_mask]
            i_instance_centroid = i_coord.mean(0)
            dist = np.linalg.norm(i_center_pred[:, :2] - i_instance_centroid[:2], axis=1)
            center_radius[i_mask] = dist
        return center_pred, center_radius

    def test(self):
        infer_once = False
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result", "inst_pred")
        make_dirs(save_path)

        f_csv = open(os.path.join(save_path, "inst_res_total.csv"), "w")
        # f_csv.write("Data Name,AP,AP50\n")

        scenes = []
        total_infer_time = 0
        all_data_avg_ap = []
        all_data_std_ap = []
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            instance = data_dict.pop("instance")
            data_name = data_dict['name']
            coord = data_dict['coord']

            # only inference
            # infer = (instance < 0).all()

            best_output = None
            best_one_ap_scores = None
            best_one_scene = None
            best_ap = -1
            total_ap = []
            # best_ap50 = -1
            # total_ap50 = []
            fragment_infer_time = 0
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
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
                    part_output_dict['pred_sem'] = F.softmax(part_output_dict["seg_logits"], -1).max(1)[1][input_dict["inverse"].cpu()]

                    # instance proposal
                    # output_dict = self.pred_feat(pred_segment, pred_bias, coord)

                idx_model_infer_time = time.time() - end
                fragment_infer_time += idx_model_infer_time

                clusters_np = part_output_dict['pred_masks'].cpu().numpy()
                pred_sem = part_output_dict['pred_sem'].cpu().numpy()
                # center_np = part_output_dict['bias_pred'].cpu().numpy() + coord
                new_clusters, new_semantic_pred, cluster_semantic_id = post_process(clusters_np, pred_sem, coord,
                                                                                    k=8,
                                                                                    npoint_th=self.cluster_propose_points,
                                                                                    eps=0.01,
                                                                                    min_points=20)
                if new_clusters is not None:
                    new_pred_sem = np.ones_like(pred_sem) * -1
                    new_pred_inst = np.ones_like(pred_sem) * -1
                    for i in range(new_clusters.shape[0]):
                        i_mask = new_clusters[i] == 1
                        new_pred_inst[i_mask] = i
                        new_pred_sem[i_mask] = cluster_semantic_id[i]
                    part_output_dict['pred_masks'] = new_clusters
                    part_output_dict['pred_classes'] = cluster_semantic_id.astype(int)
                    part_output_dict['pred_scores'] = np.ones_like(cluster_semantic_id)

                gt_instances, pred_instance = self.associate_instances(
                    part_output_dict, segment, instance
                )
                one_scene = dict(gt=gt_instances, pred=pred_instance)
                # scenes.append(one_scene)

                one_ap_scores = self.evaluate_matches([one_scene])
                one_all_ap = one_ap_scores["all_ap"]
                one_all_ap_50 = one_ap_scores["all_ap_50%"]
                one_all_ap_25 = one_ap_scores["all_ap_25%"]
                # total_ap50.append(one_all_ap_50)
                total_ap.append(one_all_ap)
                if one_all_ap > best_ap or infer_once:
                    best_output = part_output_dict
                    best_ap = one_all_ap
                    best_one_ap_scores = one_ap_scores
                    best_one_scene = one_scene

                logger.info(
                    "Test: {}/{}-{}, item: {}/{} mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}; Point Size: {} Batch {:.3f}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name,
                        i + 1,
                        len(fragment_list),
                        one_all_ap,
                        one_all_ap_50,
                        one_all_ap_25,
                        segment.shape[0],
                        idx_model_infer_time,
                    )
                )
                if infer_once:
                    break
            scenes.append(best_one_scene)
            best_ap50 = best_one_ap_scores['all_ap_50%']
            f_csv.write(f"{data_name},{best_ap},{best_ap50}")
            for nn in best_one_ap_scores['classes']:
                f_csv.write(f",{best_one_ap_scores['classes'][nn]['ap']},{best_one_ap_scores['classes'][nn]['ap50%']}")
            f_csv.write("\n")
            # 对 'bias_pred' 进行评估
            pred_center, center_radius = self.evaluate_bias(best_output['bias_pred'].cpu().numpy(), coord, instance)

            # 保存每个分割的结果
            pred_data = np.ones((coord.shape[0], 11)) * -1
            pred_data[:, :3] = coord
            pred_data[:, 3:6] = pred_center
            pred_data[:, 6] = center_radius
            pred_data[:, 9] = segment
            pred_data[:, 10] = instance
            for instance_id, mask in enumerate(best_output['pred_masks']):
                semantic_class = best_output['pred_classes'][instance_id].item()
                pred_data[mask == 1, 7] = semantic_class
                pred_data[mask == 1, 8] = instance_id  # 实例ID从0开始
            pred_save_path = os.path.join(save_path, f"{data_name}_pred.txt")
            np.savetxt(pred_save_path, pred_data)
            # if infer:
            #     logger.info(
            #         "Test: {} [{}/{}] Point Size: {} Batch {:.3f} Save Path: {}".format(
            #             data_name,
            #             idx + 1,
            #             len(self.test_loader),
            #             coord.shape[0],
            #             fragment_infer_time / len(fragment_list),
            #             pred_save_path,
            #         )
            #     )
            #     continue
            avg_ap = np.mean(total_ap)
            std_ap = np.std(total_ap)
            all_data_avg_ap.append(avg_ap)
            all_data_std_ap.append(std_ap)
            logger.info(
                "Test: {} [{}/{}] Result mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}; avg AP50 {:.4f} std:{:.4f} Point Size: {} Batch {:.3f} Save Path: {}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    best_one_ap_scores["all_ap"],
                    best_one_ap_scores["all_ap_50%"],
                    best_one_ap_scores["all_ap_25%"],
                    avg_ap,
                    std_ap,
                    segment.shape[0],
                    fragment_infer_time / len(fragment_list),
                    pred_save_path,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        all_avg_ap = np.mean(all_data_avg_ap)
        all_std_ap = np.mean(all_data_std_ap)
        logger.info(
            "Test result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f} avg mAP {:.4f} std:{:.4f} batch_time avg: {:.3f}.".format(
                all_ap, all_ap_50, all_ap_25, all_avg_ap, all_std_ap, total_infer_time / len(self.test_loader)
            )
        )

        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
