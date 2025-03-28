"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .builder import LOSSES
from pointcept.models.utils import offset2batch, batch2offset


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            size_average=None,
            reduce=None,
            reduction="mean",
            label_smoothing=0.0,
            loss_weight=1.0,
            ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight

@LOSSES.register_module()
class OHEMCrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            size_average=None,
            reduce=None,
            reduction="none",  # 先计算所有点的损失
            label_smoothing=0.0,
            loss_weight=1.0,
            ignore_index=-1,
            top_k_ratio=0.2,  # 只优化前 20% 损失最高的点
    ):
        super(OHEMCrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.top_k_ratio = top_k_ratio
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        loss = self.loss(pred, target)  # 计算所有点的损失 (不做均值)
        num_hard_samples = int(self.top_k_ratio * loss.numel())  # 选取最难的 20% 样本
        hard_loss, _ = torch.topk(loss.view(-1), num_hard_samples)  # 取损失最高的点
        return hard_loss.mean() * self.loss_weight  # 只计算这些点的损失


@LOSSES.register_module()
class GaussCrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            size_average=None,
            reduce=None,
            label_smoothing=0.0,
            loss_weight=1.0,
            ignore_index=-1,
            sigma=0.1
    ):
        super(GaussCrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.sigma = sigma
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def compute_gaussian_weights(self, coord, segment, batch, sigma=0.1):
        """
        根据高斯分布计算权重，权重最高值在交界区域，以钟形曲线延伸。

        coord: 点云的坐标 (N, 3)，包含 x, y, z。
        segment: 点的语义标签 (N,)，0 表示地面，1 表示植株。
        batch: 每个点对应的批次索引 (N,)。
        sigma: 高斯分布的标准差，控制权重的扩散范围。
        """
        # 初始化权重向量
        weights = torch.zeros_like(coord[:, 2])  # 与 z 坐标形状相同

        # 按批次循环处理
        unique_batches = torch.unique(batch)
        for b in unique_batches:
            # 获取当前批次的点
            batch_mask = (batch == b)
            coord_batch = coord[batch_mask]
            segment_batch = segment[batch_mask]

            # 提取 z 坐标（高度信息）
            z_coords = coord_batch[:, 2]

            # 找到地面和植株的点
            ground_mask = (segment_batch == 0)  # 地面点
            plant_mask = (segment_batch == 1)  # 植株点

            # 计算地面和植株的 z 范围
            z_ground_max = z_coords[ground_mask].max() if ground_mask.any() else 0.0
            z_plant_min = z_coords[plant_mask].min() if plant_mask.any() else 0.0

            # 计算交界点（高斯分布的中心）
            mu = z_ground_max + (z_plant_min - z_ground_max) / 2

            # 计算当前批次的高斯权重
            weights_batch = torch.exp(-((z_coords - mu) ** 2) / (2 * sigma ** 2))

            # 将当前批次的权重写回到全局权重向量
            weights[batch_mask] = weights_batch

        # visualize_weights(coord[:, 2], weights)

        return weights


    def compute_gaussian_weights0(self, coord, segment, sigma=0.1):
        """
        根据高斯分布计算权重，权重最高值在交界区域，以钟形曲线延伸。

        coord: 点云的坐标 (N, 3)，包含 x, y, z。
        segment: 点的语义标签 (N,)，0 表示地面，1 表示植株。
        sigma: 高斯分布的标准差，控制权重的扩散范围。
        """
        # 提取 z 坐标（高度信息）
        z_coords = coord[:, 2]

        # 找到地面和植株的点
        ground_mask = (segment == 0)  # 地面点
        plant_mask = (segment == 1)  # 植株点

        # 计算地面和植株的 z 范围
        z_ground_max = z_coords[ground_mask].max() if ground_mask.any() else 0.0
        z_plant_min = z_coords[plant_mask].min() if plant_mask.any() else 0.0

        # 计算交界点（高斯分布的中心）
        mu = z_ground_max + (z_plant_min - z_ground_max) / 2

        # 计算高斯权重
        gaussian_weights = torch.exp(-((z_coords - mu) ** 2) / (2 * sigma ** 2))

        # 初始化类别权重
        # class_weight = self.compute_dynamic_class_weights(segment)

        # combined_weights = gaussian_weights * class_weight

        return gaussian_weights

    def forward(self, pred, input_dict):
        target = input_dict["segment"]
        coord = input_dict["coord"]
        seg_loss = self.loss(pred, target) * self.loss_weight
        offset = input_dict["offset"]
        batch = offset2batch(offset)
        weights = self.compute_gaussian_weights(coord, target, batch, sigma=self.sigma)
        # weights = self.compute_gaussian_weights(coord, target, sigma=self.sigma)
        weighted_loss = (seg_loss * weights).mean()  # 加权平均
        return weighted_loss


@LOSSES.register_module()
class GaussCrossEntropyLoss0(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
        sigma_left=0.1,
        sigma_right=0.4,
        clamp_factor=2.0,
        min_val=0.1,
        # ----------------  新增 Focal Loss 的超参  ----------------
        focal_gamma=2.0,
        focal_alpha=1.0,
        use_focal=False
    ):
        """
        weight: class weight
        size_average, reduce: 这两个 PyTorch 已经不建议直接使用，统一用 reduction='none'/'mean'/'sum'
        label_smoothing: label smoothing
        loss_weight: 整体损失的一个缩放因子
        ignore_index: 忽略某个类别
        sigma: 你原本使用的高斯 sigma
        focal_gamma: Focal loss 的 gamma 超参
        focal_alpha: Focal loss 的 alpha 超参
        use_focal: 是否启用 Focal Loss 思想
        """
        super(GaussCrossEntropyLoss0, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.sigma_left=sigma_left
        self.sigma_right=sigma_right
        self.clamp_factor=clamp_factor
        self.min_val=min_val

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.use_focal = use_focal

        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction="none",  # 必须为 none，才能获取到每个样本的 loss
            label_smoothing=label_smoothing,
        )

    def compute_asymmetric_gaussian_weights_with_clamp(
        self,
        coord,
        segment,
        batch,
        sigma_left=0.1,
        sigma_right=0.4,
        clamp_factor=2.0,  # 超过多少倍 sigma 就进行下限裁剪
        min_val=0.1        # 裁剪到的固定最小值
    ):
        """
        coord:   点云的坐标 (N, 3)
        segment: 点的语义标签 (N,)
        batch:   每个点对应的批次索引 (N,)

        sigma_left:  mu 左侧的标准差
        sigma_right: mu 右侧的标准差
        clamp_factor: 超过 clamp_factor*sigma_right 就进行裁剪
        min_val: 裁剪到的最小值

        返回:
        weights: 每个点的权重 (N,)
        """

        weights = torch.zeros_like(coord[:, 2])
        unique_batches = torch.unique(batch)

        for b in unique_batches:
            # 取出该 batch 的全局下标
            batch_mask = (batch == b)
            indices_batch = batch_mask.nonzero(as_tuple=True)[0]

            # 拿到该 batch 对应的坐标/标签
            coord_batch = coord[indices_batch]
            segment_batch = segment[indices_batch]

            z_coords = coord_batch[:, 2]

            # 找到地面和植株的 z 范围
            z_ground_max = z_coords[segment_batch == 0].max() if (segment_batch == 0).any() else z_coords.min()
            z_plant_min = z_coords[segment_batch == 1].min() if (segment_batch == 1).any() else z_coords.max()

            # 计算交界高度 mu
            mu = z_ground_max + (z_plant_min - z_ground_max) / 2

            # 分别计算左侧和右侧的布尔掩码（只针对本 batch 的局部坐标）
            left_mask_local = (z_coords <= mu)
            right_mask_local = (z_coords > mu)

            # 分别计算局部权重
            weights_left = torch.exp(-((z_coords[left_mask_local] - mu) ** 2) / (2 * sigma_left ** 2))
            weights_right = torch.exp(-((z_coords[right_mask_local] - mu) ** 2) / (2 * sigma_right ** 2))

            # 1) 找到右侧哪些点超出了 clamp_factor * sigma_right
            distance_right = z_coords[right_mask_local] - mu  # > 0，因为 right_mask_local
            too_far_mask = distance_right > (clamp_factor * sigma_right)

            # 2) 对超出阈值的点，将权重裁剪到一个固定值 min_val
            weights_right = torch.where(
                too_far_mask,
                torch.full_like(weights_right, min_val),
                weights_right
            )

            # 将局部权重写回到全局 weights
            weights[indices_batch[left_mask_local]] = weights_left
            weights[indices_batch[right_mask_local]] = weights_right

        # 可视化/调试
        # visualize_weights(coord[:, 2], weights)

        return weights

    def forward(self, pred, input_dict):
        """
        pred: 模型输出，通常 shape = (N, num_classes) 或者 (B, C, H, W) ...
        input_dict: 包含 target(语义标签), coord(坐标), offset/batch 等信息
        """
        target = input_dict["segment"]    # shape = (N,)
        coord = input_dict["coord"]       # shape = (N, 3)
        offset = input_dict["offset"]     # 用于构造 batch 索引
        batch = offset2batch(offset)      # shape = (N,)

        # 1) 计算基础的 seg_loss（未做平均）
        #    由于 self.loss 是 CrossEntropyLoss(..., reduction='none'),
        #    seg_loss 形状为 (N,)
        seg_loss = self.loss(pred, target)  # 每个点对应一个 loss 值
        seg_loss = seg_loss * self.loss_weight

        # 2) 如果启用 Focal Loss，则在 seg_loss 上再乘以 focal weight
        #    focal: (1 - p_t)^gamma
        if self.use_focal:
            # 计算模型对正确类别的预测概率 p_t
            with torch.no_grad():
                # softmax 后，gather 正确类别的概率
                probs = F.softmax(pred, dim=1)                   # shape: [N, C]
                p_t = probs.gather(dim=1, index=target.unsqueeze(1))
                p_t = p_t.squeeze(1)                             # shape: [N]

            # focal weight = alpha * (1 - p_t)^gamma
            focal_weight = self.focal_alpha * (1.0 - p_t).pow(self.focal_gamma)
            seg_loss = seg_loss * focal_weight  # 与基础交叉熵逐元素相乘

        # 3) 计算你的不对称高斯权重
        weights = self.compute_asymmetric_gaussian_weights_with_clamp(
            coord,
            target,
            batch,
            sigma_left=self.sigma_left,
            sigma_right=self.sigma_right,
            clamp_factor=self.clamp_factor,
            min_val=self.min_val
        )  # shape: (N,)

        # 4) 结合高斯权重
        #    seg_loss 形状 (N,), weights 形状 (N,)
        weighted_loss = seg_loss * weights

        # 5) 做平均或求和
        final_loss = weighted_loss.mean()

        return final_loss



@LOSSES.register_module()
class GaussClassCrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            size_average=None,
            reduce=None,
            label_smoothing=0.0,
            loss_weight=1.0,
            ignore_index=-1,
            sigma=0.1
    ):
        super(GaussClassCrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.sigma = sigma
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def compute_dynamic_class_weights(self, segment):
        """
        动态计算每个批次的类别权重。

        segment_batch: 当前批次的语义标签 (N,)。
        返回: 类别权重字典。
        """
        unique_classes, counts = torch.unique(segment, return_counts=True)
        total_samples = segment.numel()
        class_weights = {int(cls): total_samples / count.item() for cls, count in zip(unique_classes, counts)}
        return class_weights

    def compute_gaussian_weights(self, coord, segment, sigma=0.1):
        """
        根据高斯分布计算权重，权重最高值在交界区域，以钟形曲线延伸。

        coord: 点云的坐标 (N, 3)，包含 x, y, z。
        segment: 点的语义标签 (N,)，0 表示地面，1 表示植株。
        sigma: 高斯分布的标准差，控制权重的扩散范围。
        """
        # 提取 z 坐标（高度信息）
        z_coords = coord[:, 2]

        # 找到地面和植株的点
        ground_mask = (segment == 0)  # 地面点
        plant_mask = (segment == 1)  # 植株点

        # 计算地面和植株的 z 范围
        z_ground_max = z_coords[ground_mask].max() if ground_mask.any() else 0.0
        z_plant_min = z_coords[plant_mask].min() if plant_mask.any() else 0.0

        # 计算交界点（高斯分布的中心）
        mu = z_ground_max + (z_plant_min - z_ground_max) / 2

        # 计算高斯权重
        gaussian_weights = torch.exp(-((z_coords - mu) ** 2) / (2 * sigma ** 2))

        # 计算动态类别权重
        class_weight_dict = self.compute_dynamic_class_weights(segment)

        # 为每个点分配类别权重
        class_weights = torch.tensor([class_weight_dict[int(cls)] for cls in segment.cpu()],
                                     dtype=torch.float32, device=coord.device)

        # 结合高斯权重和类别权重
        combined_weights = gaussian_weights * class_weights

        return combined_weights

    def forward(self, pred, input_dict):
        target = input_dict["segment"]
        coord = input_dict["coord"]
        seg_loss = self.loss(pred, target) * self.loss_weight
        weights = self.compute_gaussian_weights(coord, target, sigma=self.sigma)
        weighted_loss = (seg_loss * weights).mean()  # 加权平均
        return weighted_loss


def skew_normal_pdf(x, xi, omega, alpha):
    """
    用 PyTorch 实现支持 Tensor 的偏态正态分布的概率密度函数。

    x: 输入数据 (Tensor)。
    xi: 位置参数（偏态分布的中心）。
    omega: 尺度参数（控制宽度）。
    alpha: 偏态参数（控制分布的不对称性）。
    """
    # 计算标准化 z 值
    z = (x - xi) / omega

    sqrt_2pi = torch.sqrt(torch.tensor(2 * torch.pi, dtype=torch.float32, device=x.device))

    # 计算标准正态分布的 PDF 和 CDF
    pdf = torch.exp(-0.5 * z**2) / (sqrt_2pi * omega)
    cdf = 0.5 * (1 + torch.erf(alpha * z / torch.sqrt(torch.tensor(2.0))))

    # 偏态分布的 PDF
    skew_pdf = 2 * pdf * cdf
    return skew_pdf



def visualize_weights(z_coords, weights):
    import matplotlib.pyplot as plt
    """
    可视化权重分布
    z_coords: 点云的 z 坐标 (N,)
    weights: 对应的权重 (N,)
    """
    # 将数据按 z 坐标排序（便于可视化）
    sorted_indices = torch.argsort(z_coords)
    z_coords_sorted = z_coords[sorted_indices]
    weights_sorted = weights[sorted_indices]

    # 绘制曲线
    plt.figure(figsize=(8, 6))
    plt.plot(z_coords_sorted.cpu().numpy(), weights_sorted.cpu().numpy(), label='Gaussian Weights', color='blue')
    plt.xlabel('Height (z)', fontsize=14)
    plt.ylabel('Weight', fontsize=14)
    plt.title('Gaussian Weight Distribution', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

@LOSSES.register_module()
class SkewedCrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            size_average=None,
            reduce=None,
            label_smoothing=0.0,
            loss_weight=1.0,
            ignore_index=-1,
            sigma=0.1,
            alpha=0.0  # 偏态参数
    ):
        super(SkewedCrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.sigma = sigma
        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def compute_skew_weights(self, coord, segment, sigma=0.1, alpha=0.0):
        """
        根据偏态分布计算权重，权重最高值在交界区域，以偏态分布延伸。

        coord: 点云的坐标 (N, 3)，包含 x, y, z。
        segment: 点的语义标签 (N,)，0 表示地面，1 表示植株。
        sigma: 偏态分布的尺度参数，控制宽度。
        alpha: 偏态参数，控制分布的偏移。
        """
        # 提取 z 坐标（高度信息）
        z_coords = coord[:, 2]

        # 找到地面和植株的点
        ground_mask = (segment == 0)  # 地面点
        plant_mask = (segment == 1)  # 植株点

        # 计算地面和植株的 z 范围
        z_ground_max = z_coords[ground_mask].max() if ground_mask.any() else 0.0
        z_plant_min = z_coords[plant_mask].min() if plant_mask.any() else 0.0

        # 计算交界点（偏态分布的中心）
        mu = z_ground_max + (z_plant_min - z_ground_max) / 2

        # 计算偏态权重
        weights = skew_normal_pdf(z_coords, xi=mu, omega=sigma, alpha=alpha)

        visualize_weights(coord[:, 2], weights)

        return weights

    def forward(self, pred, input_dict):
        target = input_dict["segment"]
        coord = input_dict["coord"]
        seg_loss = self.loss(pred, target) * self.loss_weight
        weights = self.compute_skew_weights(coord, target, sigma=self.sigma, alpha=self.alpha)
        weighted_loss = (seg_loss * weights).mean()  # 加权平均
        return weighted_loss



@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
            self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
                F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                        torch.sum(
                            pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                        )
                        + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class MaskInstanceSegCriterion(nn.Module):
    def __init__(
            self,
            num_class,
            loss_weight=[1.0, 1.0, 1.0, 1.0],
            cost_weight=(1.0, 1.0, 1.0),
            ignore_index=-1):
        super(MaskInstanceSegCriterion, self).__init__()
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.matcher = HungarianMatcher(cost_weight=cost_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

    def forward(self, pred, target, **kwargs):
        instance = target['instance']
        segment = target['segment']
        offset = target['offset']

        gt_masks, gt_labels = get_insts_b(instance, segment, offset)
        target['gt_masks'] = gt_masks
        target['gt_labels'] = gt_labels

        [pred_classes, classes_offset] = pred['out_classes']
        [pred_scores, _] = pred['out_scores']
        [pred_masks, _] = pred['out_masks']

        offset = target['offset']

        point_batch = offset2batch(offset)
        batch_size = point_batch[-1] + 1
        classes_batch = offset2batch(classes_offset)

        device = offset.device

        indices = self.matcher(pred, target)

        idx = self._get_src_permutation_idx(indices)

        # class loss
        class_loss = torch.tensor(0, device=device, dtype=torch.float32)
        score_loss = torch.tensor(0, device=device, dtype=torch.float32)
        mask_bce_loss = torch.tensor(0, device=device, dtype=torch.float32)
        mask_dice_loss = torch.tensor(0, device=device, dtype=torch.float32)
        for b in range(batch_size):
            b_class_mask = classes_batch == b
            b_idx_mask = idx[0] == b
            b_tgt_class_o = gt_labels[b][indices[b][1]]

            b_pred_classes = pred_classes[b_class_mask]
            b_tgt_class = torch.full(
                [b_pred_classes.shape[0]],
                self.num_class - 1,
                dtype=torch.int64,
                device=device,
            )
            b_tgt_class[idx[1][b_idx_mask]] = b_tgt_class_o

            b_point_mask = point_batch == b
            b_pred_mask = pred_masks[b_point_mask].T[idx[1][b_idx_mask]]
            b_tgt_mask = gt_masks[b]

            with torch.no_grad():
                b_tgt_score = get_iou(b_pred_mask, b_tgt_mask).unsqueeze(1)
            b_pred_score = pred_scores[b_class_mask][idx[1][b_idx_mask]]

            class_loss += F.cross_entropy(b_pred_classes, b_tgt_class)
            score_loss += F.mse_loss(b_pred_score, b_tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(b_pred_mask, b_tgt_mask.float())
            mask_dice_loss += dice_loss(b_pred_mask, b_tgt_mask.float())

        class_loss /= batch_size
        score_loss /= batch_size
        mask_bce_loss /= batch_size
        mask_bce_loss /= batch_size
        # print('class_loss: {}'.format(class_loss))
        # tgt_class_o = torch.cat([gt_labels[idx_gt] for gt_labels, (_, idx_gt) in zip(gt_labels, indices)])
        loss = (self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss)
        # return loss
        return dict(
            loss=loss,
            class_loss=class_loss,
            score_loss=score_loss,
            mask_bce_loss=mask_bce_loss,
            mask_dice_loss=mask_dice_loss,
        )

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


@LOSSES.register_module()
class OnlyMaskInstanceSegCriterion(nn.Module):
    def __init__(
            self,
            loss_weight=[1.0, 1.0],
            cost_weight=(1.0, 1.0),
            ignore_index=-1):
        super(OnlyMaskInstanceSegCriterion, self).__init__()
        self.ignore_index = ignore_index
        self.matcher = OnlyInstanceHungarianMatcher(cost_weight=cost_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

    def forward(self, pred, target, **kwargs):
        instance = target['instance']
        segment = target['segment']
        offset = target['offset']
        loss_out = {}
        gt_masks, gt_labels = get_insts_b(instance, segment, offset)
        target['gt_masks'] = gt_masks
        target['gt_labels'] = gt_labels

        pred_masks = pred['out_masks']

        offset = target['offset']

        point_batch = offset2batch(offset)
        batch_size = point_batch[-1] + 1

        device = offset.device

        indices = self.matcher(pred, target)

        idx = self._get_src_permutation_idx(indices)

        # instance loss
        mask_bce_loss = torch.tensor(0, device=device, dtype=torch.float32)
        mask_dice_loss = torch.tensor(0, device=device, dtype=torch.float32)
        for b in range(batch_size):
            b_idx_mask = idx[0] == b

            b_point_mask = point_batch == b
            b_pred_mask = pred_masks[b_point_mask].T[idx[1][b_idx_mask]]
            b_tgt_mask = gt_masks[b]

            mask_bce_loss += F.binary_cross_entropy_with_logits(b_pred_mask, b_tgt_mask.float())
            mask_dice_loss += dice_loss(b_pred_mask, b_tgt_mask.float())

        mask_bce_loss /= batch_size
        mask_bce_loss /= batch_size
        # print('class_loss: {}'.format(class_loss))
        # tgt_class_o = torch.cat([gt_labels[idx_gt] for gt_labels, (_, idx_gt) in zip(gt_labels, indices)])
        loss = self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(reversed(pred['aux_outputs'])):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, target)
                loss += loss_i
                loss_out.update(loss_out_i)

        # return loss
        loss_out['loss'] = loss
        return loss_out

    def get_layer_loss(self, layer, aux_outputs, target):
        loss_out = {}
        instance = target['instance']
        segment = target['segment']
        target_N = instance.shape[0]

        next_cluster = aux_outputs['cluster']
        layer_mask_feat = aux_outputs['layer_mask_feat']
        aux_N = layer_mask_feat.shape[0]

        layer_pred = {}
        layer_target = {}
        if target_N == aux_N:
            gt_masks = target['gt_masks']
            # gt_labels = target['gt_labels']
            offset = target['offset']
            layer_target['gt_masks'] = gt_masks
            layer_target['offset'] = offset
            layer_pred['out_masks'] = layer_mask_feat
            # next target
            unique, cluster, counts = torch.unique(
                next_cluster, sorted=True, return_inverse=True, return_counts=True
            )
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            # next_instance = instance[idx_ptr[:-1]]
            # next_segment = segment[idx_ptr[:-1]]
            #
            layer_batch = offset2batch(offset)
            layer_batch = layer_batch[idx_ptr[:-1]]
            layer_offset = batch2offset(layer_batch)
            target[f'{layer+1}_target'] = {
                'instance': instance[idx_ptr[:-1]],
                'segment': segment[idx_ptr[:-1]],
                'offset': layer_offset
            }

        else:
            layer_instance = target[f'{layer}_target']['instance']
            layer_segment = target[f'{layer}_target']['segment']
            layer_offset = target[f'{layer}_target']['offset']
            layer_gt_masks, layer_gt_labels = get_insts_b(layer_instance, layer_segment, layer_offset)
            layer_target['gt_masks'] = layer_gt_masks
            layer_target['offset'] = layer_offset
            layer_target['gt_labels'] = layer_gt_labels
            layer_pred['out_masks'] = layer_mask_feat

            # next target
            unique, cluster, counts = torch.unique(
                next_cluster, sorted=True, return_inverse=True, return_counts=True
            )
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            # next_instance = instance[idx_ptr[:-1]]
            # next_segment = segment[idx_ptr[:-1]]
            #
            layer_batch = offset2batch(layer_offset)
            layer_batch = layer_batch[idx_ptr[:-1]]
            layer_offset = batch2offset(layer_batch)
            target[f'{layer + 1}_target'] = {
                'instance': instance[idx_ptr[:-1]],
                'segment': segment[idx_ptr[:-1]],
                'offset': layer_offset
            }


        layer_offset = layer_target['offset']
        layer_gt_masks = layer_target['gt_masks']
        layer_pred_masks = layer_pred['out_masks']

        point_batch = offset2batch(layer_offset)
        batch_size = point_batch[-1] + 1
        device = layer_offset.device

        indices = self.matcher(layer_pred, layer_target)

        idx = self._get_src_permutation_idx(indices)

        # instance loss
        mask_bce_loss = torch.tensor(0, device=device, dtype=torch.float32)
        mask_dice_loss = torch.tensor(0, device=device, dtype=torch.float32)
        for b in range(batch_size):
            b_idx_mask = idx[0] == b

            b_point_mask = point_batch == b
            b_pred_mask = layer_pred_masks[b_point_mask].T[idx[1][b_idx_mask]]
            b_tgt_mask = layer_gt_masks[b]

            mask_bce_loss += F.binary_cross_entropy_with_logits(b_pred_mask, b_tgt_mask.float())
            mask_dice_loss += dice_loss(b_pred_mask, b_tgt_mask.float())

        mask_bce_loss /= batch_size
        mask_bce_loss /= batch_size
        # print('class_loss: {}'.format(class_loss))
        # tgt_class_o = torch.cat([gt_labels[idx_gt] for gt_labels, (_, idx_gt) in zip(gt_labels, indices)])
        loss = self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}

        return loss, loss_out

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))
    """
    问题出在torch.einsum('nc,mc->nm', neg, (1 - targets))，导致的 inf，采用 focalloss 的方式优化 batch_sigmoid_bce_loss
    """
    return loss / N


@torch.jit.script
def batch_sigmoid_bce_loss_scale(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    scale = 0.5

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none') * scale
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none') * scale

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))
    """
    问题出在torch.einsum('nc,mc->nm', neg, (1 - targets))，导致的 inf，采用 focalloss 的方式优化 batch_sigmoid_bce_loss
    """
    return loss / N


# @torch.jit.script
def batch_sigmoid_bce_focal_loss1(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: 模型的输出，形状为 (num_queries, N)，N 是类别数目
        targets: 真实标签，形状为 (num_inst, N)，与 inputs 的 N 相同
        alpha: 类别权重，用于缓解类别不平衡
        gamma: 调节因子，用于降低易分类样本的权重
        reduction: 损失的降维方式，默认为 'none'
    Returns:
        Focal Loss 计算结果
    """
    alpha = 0.9
    gamma = 2.0
    N = inputs.shape[1]
    # 计算正类和负类的基本二元交叉熵损失
    pos_loss_base = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg_loss_base = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    # 计算调制因子
    pos_prob = torch.sigmoid(inputs)
    # neg_prob = 1 - pos_prob
    pos_modulation = (1 - pos_prob) ** gamma
    neg_modulation = (pos_prob) ** gamma

    # 应用 Focal Loss 思想调整损失
    epsilon = 1e-8
    pos_loss = alpha * pos_modulation * (pos_loss_base + epsilon)
    neg_loss = (1 - alpha) * neg_modulation * (neg_loss_base + epsilon)
    # pos_loss = alpha * pos_modulation * pos_loss_base
    # neg_loss = (1 - alpha) * neg_modulation * neg_loss_base

    pos_loss1 = torch.einsum('nc,mc->nm', pos_loss, targets)
    neg_loss1 = torch.einsum('nc,mc->nm', neg_loss, (1 - targets))
    print(pos_loss1, neg_loss1)
    # 使用 einsum 应用目标调整，并合并正类和负类损失
    loss = pos_loss1 + neg_loss1
    return loss / N


@torch.jit.script
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss.mean()


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss


def label_smoothing_fuc(labels, epsilon=0.1, classes=10):
    """
    对给定的标签应用标签平滑。

    参数:
        labels (torch.Tensor): one-hot 编码的标签，形状为 [classes, N]。
        epsilon (float): 平滑参数，决定平滑的强度。
        classes (int): 类别总数。

    返回:
        平滑后的标签 (torch.Tensor)。
    """
    # 确保 labels 是 one-hot 编码
    assert labels.dim() == 2 and labels.size(0) == classes
    # 计算平滑后的标签
    smoothed_labels = labels * (1 - epsilon) + (epsilon / classes)
    return smoothed_labels


def get_insts(instance_label, semantic_label, label_smoothing=True):
    """
    解决可能实例不连续的问题
    """
    device = instance_label.device
    num_insts = instance_label.max().item() + 1

    label_ids, masks = [], []

    for i in range(num_insts):
        i_inst_mask = instance_label == i
        if i_inst_mask.sum() == 0:
            continue  # 跳过不存在的实例
        sem_id = semantic_label[i_inst_mask].unique()
        if len(sem_id) > 1:
            # print(f"实例 {i} 有多个语义标签：{sem_id}")
            sem_id = sem_id[0]  # 假设选择第一个标签，需要根据实际情况调整
        label_ids.append(sem_id.item())
        masks.append(i_inst_mask)

    if len(masks) == 0:
        return None, None  # 没有有效的实例

    masks = torch.stack(masks).int()  # 将布尔掩码列表堆叠成张量，并转换为整数
    label_ids = torch.tensor(label_ids, device=device)

    # label_smoothing
    if label_smoothing:
        masks = label_smoothing_fuc(masks, classes=num_insts)
    return masks, label_ids


def get_insts_b(instance, segment, offset, label_smoothing=False):
    # instance = data_dict['instance']
    # segment = data_dict['segment']
    #
    # offset = data_dict['offset']

    point_batch = offset2batch(offset)
    batch_size = point_batch[-1] + 1

    gt_masks, gt_labels = [], []
    for b in range(batch_size):
        b_point_mask = point_batch == b
        b_gt_masks, b_gt_labels = get_insts(instance[b_point_mask], segment[b_point_mask],
                                            label_smoothing=label_smoothing)
        gt_masks.append(b_gt_masks)
        gt_labels.append(b_gt_labels)

    return gt_masks, gt_labels


class OnlyInstanceHungarianMatcher(nn.Module):
    def __init__(self, cost_weight):
        super(OnlyInstanceHungarianMatcher, self).__init__()
        self.cost_weight = torch.tensor(cost_weight)

    @torch.no_grad()
    def forward(self, pred, target):
        pred_masks = pred['out_masks']

        # instance = target['instance']
        # segment = target['segment']

        gt_masks = target['gt_masks']

        offset = target['offset']
        # num_queries = pred_masks.size(1)

        point_batch = offset2batch(offset)
        batch_size = point_batch[-1] + 1

        indices = []
        for b in range(batch_size):
            b_point_mask = point_batch == b

            # gt_masks, gt_labels = get_insts(instance[b_point_mask], segment[b_point_mask])
            b_gt_mask = gt_masks[b]

            pred_mask = pred_masks[b_point_mask].T  # [200, N]
            tgt_mask = b_gt_mask  # [num_ins, N]

            # cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            # cost_mask = batch_sigmoid_bce_focal_loss(pred_mask, tgt_mask.float())
            cost_mask = batch_sigmoid_bce_loss_scale(pred_mask, tgt_mask.float())

            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcher(nn.Module):
    def __init__(self, cost_weight):
        super(HungarianMatcher, self).__init__()

        self.cost_weight = torch.tensor(cost_weight)

    @torch.no_grad()
    def forward(self, pred, target):
        [pred_masks, _] = pred['out_masks']
        [pred_logits, logit_offset] = pred['out_classes']

        # instance = target['instance']
        # segment = target['segment']

        gt_labels = target['gt_labels']
        gt_masks = target['gt_masks']

        offset = target['offset']
        # num_queries = pred_masks.size(1)

        point_batch = offset2batch(offset)
        batch_size = point_batch[-1] + 1
        logit_batch = offset2batch(logit_offset)

        indices = []
        for b in range(batch_size):
            b_logit_mask = logit_batch == b
            b_point_mask = point_batch == b

            # gt_masks, gt_labels = get_insts(instance[b_point_mask], segment[b_point_mask])
            b_gt_mask = gt_masks[b]
            b_gt_labels = gt_labels[b]

            pred_prob = pred_logits[b_logit_mask].softmax(-1)
            tgt_idx = b_gt_labels
            cost_class = -pred_prob[:, tgt_idx]

            pred_mask = pred_masks[b_point_mask].T  # [200, N]
            tgt_mask = b_gt_mask  # [num_ins, N]

            # cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            # cost_mask = batch_sigmoid_bce_focal_loss(pred_mask, tgt_mask.float())
            cost_mask = batch_sigmoid_bce_loss_scale(pred_mask, tgt_mask.float())

            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[0] * cost_class + self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
