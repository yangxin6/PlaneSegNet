"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.models.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)

        self.conv3 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn3 = norm_fn(embed_channels)

        self.stride = stride

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_channels * 3, embed_channels),  # 从输入维度到隐藏层
            nn.LeakyReLU(),  # 激活函数
            nn.Linear(embed_channels, embed_channels)  # 从隐藏层到输出维度
        )

    def plane_features(self, x):
        indices = x.indices  # [num_points, 4]，存储每个点的 [batch_id, x, y, z]
        features = x.features  # [num_points, num_features]，存储每个点的特征

        # 提取 XZ 平面坐标并去重
        xz_coords = indices[:, [0, 1, 3]]  # 保留 batch_id, x, z 坐标
        xz_unique_coords, xz_inverse_indices = torch.unique(xz_coords, dim=0, return_inverse=True)  # 唯一坐标和映射

        # 对 XZ 平面特征进行聚合
        xz_aggregated_features = scatter(features, xz_inverse_indices, dim=0, reduce='mean')  # 使用 'mean', 'sum', 'max' 等方式聚合

        # 提取 YZ 平面坐标并去重
        yz_coords = indices[:, [0, 2, 3]]  # 保留 batch_id, y, z 坐标
        yz_unique_coords, yz_inverse_indices = torch.unique(yz_coords, dim=0, return_inverse=True)

        # 对 YZ 平面特征进行聚合
        yz_aggregated_features = scatter(features, yz_inverse_indices, dim=0, reduce='mean')  # 同样使用 'mean'

        aligned_xz_features = xz_aggregated_features[xz_inverse_indices]  # 按照索引映射回去
        aligned_yz_features = yz_aggregated_features[yz_inverse_indices]

        return aligned_xz_features, aligned_yz_features

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        aligned_xz_features, aligned_yz_features = self.plane_features(out)

        out1 = self.conv3(out)
        out1 = out1.replace_feature(self.bn3(out1.features))
        out1 = out1.replace_feature(self.relu(out1.features))

        # 融合平面特征与卷积特征
        concatenated_features = torch.cat([out1.features, aligned_xz_features, aligned_yz_features], dim=1)
        fused_features = self.fusion_mlp(concatenated_features)

        # 加入残差
        out = out.replace_feature(fused_features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


@MODELS.register_module("SpVUNet-v1m2")
class SpVUNetBase(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=False,
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.cls_mode = cls_mode

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.LeakyReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.cls_mode else None

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.LeakyReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            if not self.cls_mode:
                # decode num_stages
                self.up.append(
                    spconv.SparseSequential(
                        spconv.SparseInverseConv3d(
                            channels[len(channels) - s - 2],
                            dec_channels,
                            kernel_size=2,
                            bias=False,
                            indice_key=f"spconv{s + 1}",
                        ),
                        norm_fn(dec_channels),
                        nn.LeakyReLU(),
                    )
                )
                self.dec.append(
                    spconv.SparseSequential(
                        OrderedDict(
                            [
                                (
                                    (
                                        f"block{i}",
                                        block(
                                            dec_channels + enc_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                    if i == 0
                                    else (
                                        f"block{i}",
                                        block(
                                            dec_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                )
                                for i in range(layers[len(channels) - s - 1])
                            ]
                        )
                    )
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        final_in_channels = (
            channels[-1] if not self.cls_mode else channels[self.num_stages - 1]
        )
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]

        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        if not self.cls_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)

        x = self.final(x)
        if self.cls_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )
        return x.features

