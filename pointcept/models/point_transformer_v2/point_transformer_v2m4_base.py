"""
Point Transformer V2 Mode 2 (recommend)

Disable Grouped Linear

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath
import pointops

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(
            self,
            embed_channels,
            groups,
            attn_drop_rate=0.0,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class Block(nn.Module):
    def __init__(
            self,
            embed_channels,
            groups,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            enable_checkpoint=False,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = (
            self.attn(feat, coord, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat, coord, reference_index)
        )
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(
            self,
            depth,
            embed_channels,
            groups,
            neighbours=16,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            enable_checkpoint=False,
    ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class FPSPool(nn.Module):
    def __init__(self, in_channels, out_channels, num_samples, bias=False):
        super(FPSPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_samples = num_samples

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))

        new_coords, new_feats, new_offsets = [], [], [0]  # 初始化new_offsets包含起始点0
        for b in range(batch.max().item() + 1):  # 确保batch.max()为int
            b_mask = batch == b
            b_coord = coord[b_mask]
            b_feat = feat[b_mask]
            if b_coord.size(0) < self.num_samples:
                # 如果当前批次的点数少于num_samples，采用全部点
                b_fps_indices = torch.arange(b_coord.size(0), device=coord.device)
            else:
                # 否则，使用FPS采样
                b_fps_indices = self.farthest_point_sample(b_coord, self.num_samples)
            new_coords.append(b_coord[b_fps_indices])
            new_feats.append(b_feat[b_fps_indices])
            new_offsets.append(new_offsets[-1] + b_fps_indices.size(0))

        new_coords = torch.cat(new_coords, dim=0)
        new_feats = torch.cat(new_feats, dim=0)
        new_offsets.pop(0)
        new_offsets = torch.tensor(new_offsets, dtype=torch.long)

        return [new_coords, new_feats, new_offsets]

    def farthest_point_sample(self, xyz, npoint):
        device = xyz.device
        N, _ = xyz.shape
        centroids = torch.zeros(npoint, dtype=torch.long, device=device)
        distance = torch.ones(N, device=device) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)

        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest]
            dist = torch.sum((xyz - centroid) ** 2, dim=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = (
            segment_csr(
                coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            )
            if start is None
            else start
        )
        cluster = voxel_grid(
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
        )
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            bias=True,
            skip=True,
            backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(
                coord, skip_coord, self.proj(feat), offset, skip_offset
            )
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class Encoder(nn.Module):
    def __init__(
            self,
            depth,
            in_channels,
            embed_channels,
            groups,
            grid_size=None,
            neighbours=16,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=None,
            drop_path_rate=None,
            enable_checkpoint=False,
    ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            embed_channels,
            groups,
            depth,
            neighbours=16,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=None,
            drop_path_rate=None,
            enable_checkpoint=False,
            unpool_backend="map",
    ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(
            self,
            depth,
            in_channels,
            embed_channels,
            groups,
            neighbours=16,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            enable_checkpoint=False,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


class InstanceMaskModule(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 enable_checkpoint=False,
                 ):
        super(InstanceMaskModule, self).__init__()

        self.neighbours = neighbours
        self.instance_block = Block(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset = points
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        points = self.instance_block(points, reference_index)

        return points


@MODELS.register_module("PT-v2m4")
class PointTransformerV2(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            patch_embed_depth=1,
            patch_embed_channels=48,
            patch_embed_groups=6,
            patch_embed_neighbours=8,
            enc_depths=(2, 2, 6, 2),
            enc_channels=(96, 192, 384, 512),
            enc_groups=(12, 24, 48, 64),
            enc_neighbours=(16, 16, 16, 16),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(48, 96, 192, 384),
            dec_groups=(6, 12, 24, 48),
            dec_neighbours=(16, 16, 16, 16),
            grid_sizes=(0.06, 0.12, 0.24, 0.48),
            attn_qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0,
            enable_checkpoint=False,
            unpool_backend="map",
            topk_insts=100,
            score_thr=0.0,
            npoint_thr=1,
    ):
        super(PointTransformerV2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
        )

        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                               sum(enc_depths[:i]): sum(enc_depths[: i + 1])
                               ],
                enable_checkpoint=enable_checkpoint,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[
                               sum(dec_depths[:i]): sum(dec_depths[: i + 1])
                               ],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)

        seg_query_num = 200
        self.fps_pool = FPSPool(in_channels=enc_channels[-2], out_channels=enc_channels[-1], num_samples=seg_query_num)

        d_model = 512

        self.feat_proj = nn.Sequential(
            nn.Linear(dec_channels[0], d_model),
            PointBatchNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.query_proj = nn.Sequential(
            nn.Linear(enc_channels[-1], d_model),
            PointBatchNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.decoder_norm = nn.LayerNorm(d_model)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            PointBatchNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # self.class_embed_head = nn.Linear(d_model, self.num_classes)
        self.class_embed_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            PointBatchNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.num_classes)
        )
        self.out_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            PointBatchNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.topk_insts = topk_insts
        self.score_thr = score_thr
        self.npoint_thr = npoint_thr

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage

        query_init_points = skips[-2][0]

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        # decoder_points = []
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
            # decoder_points.append(points)

        instance_query_points = self.fps_pool(query_init_points)

        # last stage MM
        out_class, out_scores, out_masks = self.mask_module(instance_query_points, points)
        return_dict = dict(
            out_classes=out_class,
            out_scores=out_scores,
            out_masks=out_masks,
        )
        if not self.training:
            pred_dict = self.predict_by_feat(return_dict)
            return_dict.update(pred_dict)
        return return_dict

    def mask_module(self, query_points, feat_points):

        _, query_feat, query_offset = query_points
        _, point_feat, point_offset = feat_points

        point_feat = self.feat_proj(point_feat)

        query_feat = self.decoder_norm(self.query_proj(query_feat))
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
        out_score = self.out_score(query_feat)

        query_batch = offset2batch(query_offset)
        point_batch = offset2batch(point_offset)

        batch_size = query_batch[-1] + 1

        output_masks = []
        # output_masks = feat @ mask_embed.T
        for b in range(batch_size):
            b_query_mask = query_batch == b
            b_point_mask = point_batch == b

            b_output_mask = point_feat[b_point_mask] @ mask_embed[b_query_mask].T
            output_masks.append(b_output_mask)
        output_masks = torch.cat(output_masks)
        return [outputs_class, query_offset], [out_score, query_offset], [output_masks, point_offset]

    def predict_by_feat(self, outputs_dict):
        [pred_classes, classes_offset] = outputs_dict['out_classes']
        [pred_scores, _] = outputs_dict['out_scores']
        [pred_masks, point_offset] = outputs_dict['out_masks']

        point_batch = offset2batch(point_offset)
        batch_size = point_batch[-1] + 1
        classes_batch = offset2batch(classes_offset)

        scores = []
        labels = []
        masks = []

        for b in range(batch_size):
            b_class_mask = classes_batch == b
            b_point_mask = point_batch == b

            b_pred_classes = pred_classes[b_class_mask]
            b_pred_scores = pred_scores[b_class_mask]
            b_pred_masks = pred_masks[b_point_mask].T

            b_scores = F.softmax(b_pred_classes, dim=-1)
            b_scores *= b_pred_scores

            b_labels = torch.arange(
                self.num_classes, device=b_scores.device).unsqueeze(0).repeat(len(b_class_mask), 1).flatten(0, 1)
            b_scores, b_topk_idx = b_scores.flatten(0, 1).topk(self.topk_insts, sorted=False)
            b_labels = b_labels[b_topk_idx]

            b_topk_idx = torch.div(b_topk_idx, self.num_classes, rounding_mode='floor')

            b_mask_pred = b_pred_masks[b_topk_idx]
            b_mask_pred_sigmoid = b_mask_pred.sigmoid()
            # mask_pred before sigmoid()
            b_mask_pred = (b_mask_pred > 0).float()  # [n_p, M]
            b_mask_scores = (b_mask_pred_sigmoid * b_mask_pred).sum(1) / (b_mask_pred.sum(1) + 1e-6)
            b_scores = b_scores * b_mask_scores

            # score_thr
            b_score_mask = b_scores > self.score_thr
            b_scores = b_scores[b_score_mask]  # (n_p,)
            b_labels = b_labels[b_score_mask]  # (n_p,)
            b_mask_pred = b_mask_pred[b_score_mask]  # (n_p, N)

            # npoint thr
            b_mask_pointnum = b_mask_pred.sum(1)
            b_npoint_mask = b_mask_pointnum > self.npoint_thr
            b_scores = b_scores[b_npoint_mask]  # (n_p,)
            b_labels = b_labels[b_npoint_mask]  # (n_p,)
            b_mask_pred = b_mask_pred[b_npoint_mask]  # (n_p, N)

            scores.append(b_scores)
            labels.append(b_labels)
            masks.append(b_mask_pred)
        print(labels, scores, masks)
        return dict(
            pred_classes=torch.cat(labels).cpu().numpy(),
            pred_scores=torch.cat(scores).cpu().numpy(),
            pred_masks=torch.cat(masks).cpu().numpy(),
        )