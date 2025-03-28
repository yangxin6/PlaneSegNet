"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import torch

from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        # 当没有指定具体损失函数时，可能是模型内部计算损失
        if len(self.criteria) == 0:
            return pred

        # 初始化总损失
        total_loss = 0
        # 初始化用于存储各组件损失的字典
        loss_components = {}

        for c in self.criteria:
            c_loss = c(pred, target)

            # 如果 c_loss 是数字，直接累加到 total_loss
            if isinstance(c_loss, (float, int, torch.Tensor)):
                total_loss += c_loss
            # 如果 c_loss 是字典，逐项累加到 loss_components
            elif isinstance(c_loss, dict):
                for key, value in c_loss.items():
                    if key in loss_components:
                        loss_components[key] += value
                    else:
                        loss_components[key] = value
            else:
                # 对未知类型的损失值给出警告
                print(f"Warning: Unsupported loss type {type(c_loss)} returned by a criterion.")

        # 如果有多个损失组件，返回包含总损失和各组件损失的字典
        if loss_components:
            return loss_components
        # 如果只有总损失，直接返回总损失
        else:
            return total_loss


def build_criteria(cfg):
    return Criteria(cfg)
