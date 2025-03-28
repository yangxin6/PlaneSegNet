"""
3D Point Cloud Augmentation

Inspirited by chrischoy/SpatioTemporalSegmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from pointcept.utils.registry import Registry
TRANSFORMS = Registry("transforms")
np.random.seed(123456)


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class Add(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
            if "organ_segment" in data_dict.keys():
                data_dict["organ_segment"] = data_dict["organ_segment"][idx]
            if "organ_instance" in data_dict.keys():
                data_dict["organ_instance"] = data_dict["organ_instance"][idx]
            if "sp_feat" in data_dict.keys():
                data_dict["sp_feat"] = data_dict["sp_feat"][idx]
            if "boundary" in data_dict.keys():
                data_dict["boundary"] = data_dict["boundary"][idx]
        return data_dict


@TRANSFORMS.register_module()
class RandomTreeMix(object):
    def __init__(self, replace_ratio=0.3, replace_application_ratio=0.5, template_path=None):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.replace_ratio = replace_ratio
        self.replace_application_ratio = replace_application_ratio
        self.template_path = template_path

    def calculate_overlap(self, point_cloud1, point_cloud2, threshold):
        """
        Calculate the spatial overlap between two point clouds using KD-trees with tensors.

        Args:
            point_cloud1 (torch.Tensor): First point cloud, shape (N, 3).
            point_cloud2_tree (torch.Tensor): kdtree of the second point cloud, shape (M, 3).
            threshold (float): Maximum distance for a point to be considered overlapping.

        Returns:
            float: Overlap ratio between the two point clouds.
        """
        tree = cKDTree(point_cloud2)
        distances, _ = tree.query(point_cloud1, k=1, distance_upper_bound=threshold)
        valid_distances = distances != float('inf')
        num_overlap = torch.sum(torch.from_numpy(valid_distances))
        overlap_ratio = num_overlap / len(point_cloud1)
        return overlap_ratio

    def calculate_normals(self, coords, k_neighbors=10, radius=None):
        """
        计算点云的法线。

        参数:
        - coords: Nx3的NumPy数组，表示点云的坐标。
        - k_neighbors: 用于估计每个点法线的最近邻的数量。
        - radius: 搜索半径，用于估计每个点的法线。如果指定了radius，则使用半径搜索而不是k最近邻搜索。

        返回:
        - normals: Nx3的NumPy数组，表示每个点的法线向量。
        """
        # 将coords数组转换为Open3D的点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)

        # 计算法线
        if radius is not None:
            # 使用半径搜索来计算法线
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k_neighbors))
        else:
            # 使用k最近邻搜索来计算法线
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))

        # 可选：根据视点方向翻转法线
        # pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

        # 从点云对象中提取法线并转换为NumPy数组
        normals = np.asarray(pcd.normals)

        return normals

    def simple_augment_coord(self, coords):
        # 简单演示：随机平移 + 轻微旋转
        angle = np.random.uniform(-10, 10) / 180.0 * np.pi
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)

        # 只在 XY 平面做一个小旋转
        # coords shape: [M, 3]
        coords_rot = coords.copy()
        x, y = coords_rot[:, 0], coords_rot[:, 1]
        coords_rot[:, 0] = x * cos_val - y * sin_val
        coords_rot[:, 1] = x * sin_val + y * cos_val

        # 随机平移
        shift = np.random.uniform(-1, 1, size=(3,))
        coords_rot += shift

        # 应用缩放
        scale_factor = np.random.uniform(0.9, 1.1)
        coords_rot *= scale_factor
        return coords_rot

    def __call__(self, data_dict):
        if random.random() < self.replace_application_ratio:
            return data_dict

        if not self.template_path:
            return data_dict

        tem_data = np.load(self.template_path, allow_pickle=True)
        data_set = tem_data['data_set'].tolist()

        group_coord = data_dict['coord']  # [N, 3]
        group_color = data_dict['color']  # [N, 3]
        group_normal = data_dict['normal']  # [N, 3]
        group_semantic_gt = data_dict['segment']  # [N]
        group_instance_gt = data_dict['instance']  # [N]
        group_organ_semantic_gt = data_dict['organ_segment']  # [N]

        # 2. 找到场景中的所有实例 ID，并随机选取 30%
        unique_instances = np.unique(group_instance_gt)
        total_inst_count = len(unique_instances)
        replace_count = int(total_inst_count * self.replace_ratio)  # 30%

        # 随机选择 replace_count 个实例 ID
        replace_inst_ids = random.sample(unique_instances.tolist(), replace_count)

        # 3. 从 data_set 中也随机选出与替换数相同的单株树
        data_set_keys = list(data_set.keys())  # data_set 的所有 key
        selected_tree_keys = random.sample(data_set_keys, replace_count)

        # 4. 针对被替换的实例，逐个执行替换逻辑
        #    下面仅示范逻辑流程，具体的 treemix、数据增强等需根据实际功能实现
        new_coords_list = []
        new_colors_list = []
        new_normals_list = []
        new_semantic_list = []
        new_instance_list = []
        new_organ_semantic_list = []

        #   ——> 首先将未被替换的实例的点云保留
        mask_keep = np.ones_like(group_instance_gt, dtype=bool)
        for inst_id in replace_inst_ids:
            mask_keep = mask_keep & (group_instance_gt != inst_id)

        #   ——> 保留部分
        new_coords_list.append(group_coord[mask_keep])
        new_colors_list.append(group_color[mask_keep])
        new_normals_list.append(group_normal[mask_keep])
        new_semantic_list.append(group_semantic_gt[mask_keep])
        new_instance_list.append(group_instance_gt[mask_keep])
        new_organ_semantic_list.append(group_organ_semantic_gt[mask_keep])

        # 5. 替换部分：将从 data_set 中选出的单株树插入场景
        max_inst_id = group_instance_gt.max()
        current_new_inst_id = max_inst_id + 1  # 用于给插入的树分配新 ID

        for i, inst_id in enumerate(replace_inst_ids):
            # 5.1 从 data_set 中随机选到的一棵树
            tree_key = selected_tree_keys[i]
            i_data = data_set[tree_key]  # shape: [M, 12]

            i_coord = i_data[:, :3]
            i_color = i_data[:, 3:6]
            i_normal = i_data[:, 6:9]
            i_organ_semantic_gt = i_data[:, 9]
            i_semantic_gt = i_data[:, 10]
            i_instance_gt = i_data[:, 11]  # 通常为 0 或统一值

            # 应用数据增强操作
            i_coord = self.simple_augment_coord(i_coord)

            # 替换后的树实例的位置会被调整
            # 首先对齐 z 的最低值
            min_z_diff = np.min(i_coord[:, 2]) - np.min(group_coord[group_instance_gt == inst_id, 2])
            i_coord[:, 2] -= min_z_diff

            # 替换后的树实例的位置会被调整，以确保与现有树实例的位置不发生冲突。例如，将插入数据的坐标调整到与原树实例的平均位置对齐。
            offset = np.mean(i_coord[:, :2], axis=0) - np.mean(group_coord[group_instance_gt == inst_id, :2], axis=0)
            i_coord[:, :2] -= offset

            # 在插入替换的树实例之前，计算其与原数据中的树实例的重叠程度。
            overlap_ratio = self.calculate_overlap(i_coord, group_coord, threshold=0.3)
            if overlap_ratio < 0.1:
                i_instance_gt = np.full(i_instance_gt.shape, current_new_inst_id)
                current_new_inst_id += 1
                new_coords_list.append(i_coord)
                new_colors_list.append(i_color)
                new_normals_list.append(i_normal)
                new_semantic_list.append(i_semantic_gt)
                new_instance_list.append(i_instance_gt)
                new_organ_semantic_list.append(i_organ_semantic_gt)

        # 6. 拼接所有保留点和新插入点，形成最终的场景
        final_coord = np.concatenate(new_coords_list, axis=0)
        final_color = np.concatenate(new_colors_list, axis=0)
        final_normal = self.calculate_normals(final_coord)
        final_semantic = np.concatenate(new_semantic_list, axis=0)
        final_instance = np.concatenate(new_instance_list, axis=0)
        final_organ_semantic = np.concatenate(new_organ_semantic_list, axis=0)

        data_dict['coord'] = final_coord
        data_dict['color'] = final_color
        data_dict['normal'] = final_normal
        data_dict['segment'] = final_semantic
        data_dict['instance'] = final_instance
        data_dict['organ_segment'] = final_organ_semantic
        # fake organ instance
        data_dict['organ_instance'] = np.zeros_like(final_instance)
        return data_dict



@TRANSFORMS.register_module()
class RandomCropXY(object):
    def __init__(self, crop_size):
        """
        crop_size: The size of the crop in the xy plane (e.g., [width, height])
        """
        self.crop_size = np.array(crop_size)
        self.min_size = 100

    def __call__(self, data_dict):
        # Get the minimum and maximum coordinates in the xy-plane
        xy_min = data_dict["coord"][:, :2].min(axis=0)
        xy_max = data_dict["coord"][:, :2].max(axis=0)

        xy_range = xy_max - xy_min

        if np.any(xy_range < self.crop_size):
            # If the point cloud is smaller, return the original data without cropping
            return data_dict

        # Randomly choose a corner for the crop within the valid range
        crop_min = xy_min + np.random.rand(2) * (xy_max - xy_min - self.crop_size)
        crop_max = crop_min + self.crop_size

        # Create a mask for points within the cropped region in the xy-plane
        mask = np.all((data_dict["coord"][:, :2] >= crop_min) & (data_dict["coord"][:, :2] <= crop_max), axis=1)

        field_dict = ["coord", "color", "normal", "strength", "boundary", "segment", "instance", "organ_segment", "organ_instance"]
        # if "instance" in data_dict:
        #     inst = data_dict["instance"]
        #     unique_inst, inst_counts = np.unique(inst, return_counts=True)
        #     valid_inst = unique_inst[inst_counts >= 50]
        #     mask_inst = np.isin(inst, valid_inst)
        #     mask = mask & mask_inst  # Combine the crop mask with the instance mask
        # Apply the mask to all relevant fields
        for key in field_dict:
            if key in data_dict:
                data_dict[key] = data_dict[key][mask]

        # Handle special cases if needed
        if "sampled_index" in data_dict:
            mask_indices = np.where(mask)[0]
            data_dict["sampled_index"] = np.intersect1d(data_dict["sampled_index"], mask_indices)

        if "instance" in data_dict.keys():
            u_instance = np.unique(data_dict["instance"][mask])
            valid_instance = []
            for u in u_instance:
                inst_mask = data_dict["instance"][mask] == u
                inst_points = np.sum(inst_mask)
                if inst_points > self.min_size:
                    valid_instance.append(u)
            valid_mask = np.isin(data_dict["instance"][mask], valid_instance)

            for key in field_dict:
                if key in data_dict:
                    data_dict[key] = data_dict[key][valid_mask]

            # Handle special cases if needed
            if "sampled_index" in data_dict:
                mask_indices = np.where(valid_mask)[0]
                data_dict["sampled_index"] = np.intersect1d(data_dict["sampled_index"], mask_indices)

        if "instance" in data_dict:
            inst = data_dict["instance"]
            unique_inst, inst_counts = np.unique(inst, return_counts=True)
            valid_inst = unique_inst[inst_counts >= 100]
            mask_inst = np.isin(inst, valid_inst)

            for key in ["coord", "color", "normal", "strength", "boundary", "segment", "instance", "organ_segment",
                        "organ_instance"]:
                if key in data_dict:
                    data_dict[key] = data_dict[key][mask_inst]

                # Handle special cases if needed
            if "sampled_index" in data_dict:
                mask_indices = np.where(mask_inst)[0]
                data_dict["sampled_index"] = np.intersect1d(data_dict["sampled_index"], mask_indices)

        return data_dict


@TRANSFORMS.register_module()
class FilterMinInst(object):
    def __init__(self, num=100):
        """
        crop_size: The size of the crop in the xy plane (e.g., [width, height])
        """
        self.num = num

    def __call__(self, data_dict):

        if "instance" in data_dict:
            unique_inst, inst_counts = np.unique(data_dict["instance"], return_counts=True)
            valid_inst_ids = unique_inst[inst_counts >= self.num]
            mask_complete = np.isin(data_dict["instance"], valid_inst_ids)

            for key in ["coord", "color", "normal", "strength", "boundary", "segment", "instance", "organ_segment",
                        "organ_instance"]:
                if key in data_dict:
                    data_dict[key] = data_dict[key][mask_complete]
            if "sampled_index" in data_dict:
                mask_indices = np.where(mask_complete)[0]
                data_dict["sampled_index"] = np.intersect1d(data_dict["sampled_index"], mask_indices)
        return data_dict


@TRANSFORMS.register_module()
class RandomCropComXY(object):
    def __init__(self, crop_size):
        """
        crop_size: The size of the crop in the xy plane (e.g., [width, height])
        """
        self.crop_size = np.array(crop_size)

    def __call__(self, data_dict):
        # Get the minimum and maximum coordinates in the xy-plane
        xy_min = data_dict["coord"][:, :2].min(axis=0)
        xy_max = data_dict["coord"][:, :2].max(axis=0)

        xy_range = xy_max - xy_min

        if np.any(xy_range < self.crop_size):
            # If the point cloud is smaller, return the original data without cropping
            # 如果点云范围小于裁剪尺寸，过滤点数小于 100 的实例
            if "instance" in data_dict:
                unique_inst, inst_counts = np.unique(data_dict["instance"], return_counts=True)
                valid_inst_ids = unique_inst[inst_counts >= 100]
                mask_complete = np.isin(data_dict["instance"], valid_inst_ids)

                for key in ["coord", "color", "normal", "strength", "boundary", "segment", "instance", "organ_segment",
                            "organ_instance"]:
                    if key in data_dict:
                        data_dict[key] = data_dict[key][mask_complete]

                if "sampled_index" in data_dict:
                    mask_indices = np.where(mask_complete)[0]
                    data_dict["sampled_index"] = np.intersect1d(data_dict["sampled_index"], mask_indices)
            return data_dict

        # Randomly choose a corner for the crop within the valid range
        crop_min = xy_min + np.random.rand(2) * (xy_max - xy_min - self.crop_size)
        crop_max = crop_min + self.crop_size

        # Create a mask for points within the cropped region in the xy-plane
        mask = np.all((data_dict["coord"][:, :2] >= crop_min) & (data_dict["coord"][:, :2] <= crop_max), axis=1)

        if "instance" in data_dict:
            # 获取裁剪区域内的实例 ID
            cropped_inst_ids = np.unique(data_dict["instance"][mask])

            # 统计每个实例的点数量
            unique_inst, inst_counts = np.unique(data_dict["instance"], return_counts=True)

            # 过滤点数大于等于 100 的实例
            valid_inst_ids = unique_inst[inst_counts >= 100]

            # 仅保留同时在裁剪区域内且点数大于等于 100 的实例
            final_inst_ids = np.intersect1d(cropped_inst_ids, valid_inst_ids)

            # 保留完整实例：创建一个新的掩码，包含完整实例
            mask_complete = np.isin(data_dict["instance"], final_inst_ids)
        else:
            mask_complete = mask  # 如果没有实例信息，直接使用裁剪掩码

        # Apply the mask to all relevant fields
        for key in ["coord", "color", "normal", "strength", "boundary", "segment", "instance", "organ_segment", "organ_instance"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][mask_complete]

        # Handle special cases if needed
        if "sampled_index" in data_dict:
            mask_indices = np.where(mask_complete)[0]
            data_dict["sampled_index"] = np.intersect1d(data_dict["sampled_index"], mask_indices)

        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
            self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
            # import open3d as o3d
            # xyz = data_dict["coord"]
            # # 创建第一个点云对象
            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(xyz)
            #
            # o3d.visualization.draw_geometries([pcd1], )  # z轴朝上
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                                                             :, :3
                                                             ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
            value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                    fn_id == 0
                    and brightness_factor is not None
                    and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                    fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                    fn_id == 2
                    and saturation_factor is not None
                    and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
            self,
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            keys=("coord", "color", "normal", "segment"),
            return_inverse=False,
            return_grid_coord=False,
            return_min_coord=False,
            return_displacement=False,
            project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                        scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                # if "superpoint" == key:
                #     _, new_superpoints = data_dict["superpoint"][idx_unique].unique(return_inverse=True)
                #     data_dict["superpoint"] = new_superpoints
                # else:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict
        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                            scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > point_max:
                coord_p, idx_uni = np.random.rand(
                    data_dict["coord"].shape[0]
                ) * 1e-3, np.array([])
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(
                        np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2),
                        1,
                    )
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "grid_coord" in data_dict.keys():
                        data_crop_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    if "displacement" in data_dict.keys():
                        data_crop_dict["displacement"] = data_dict["displacement"][
                            idx_crop
                        ]
                    if "strength" in data_dict.keys():
                        data_crop_dict["strength"] = data_dict["strength"][idx_crop]
                    if "boundary" in data_dict.keys():
                        data_crop_dict["boundary"] = data_dict["boundary"][idx_crop]
                    if "sp_feat" in data_dict.keys():
                        data_crop_dict["sp_feat"] = data_dict["sp_feat"][idx_crop]
                    if "superpoint" in data_dict.keys():
                        # _, new_superpoints = data_dict["superpoint"][idx_crop].unique(return_inverse=True)
                        # data_crop_dict["superpoint"] = new_superpoints
                        data_crop_dict["superpoint"] = data_dict["superpoint"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(
                        1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"])
                    )
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(
                        np.concatenate((idx_uni, data_crop_dict["index"]))
                    )
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                       :point_max
                       ]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "organ_segment" in data_dict.keys():
                data_dict["organ_segment"] = data_dict["organ_segment"][idx_crop]
            if "organ_instance" in data_dict.keys():
                data_dict["organ_instance"] = data_dict["organ_instance"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "boundary" in data_dict.keys():
                data_dict["boundary"] = data_dict["boundary"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
            if "superpoint" in data_dict.keys():
                # _, new_superpoints = data_dict["superpoint"][idx_crop].unique(return_inverse=True)
                # data_dict["superpoint"] = new_superpoints
                data_dict["superpoint"] = data_dict["superpoint"][idx_crop]
        return data_dict


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][shuffle_index]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][shuffle_index]
        if "displacement" in data_dict.keys():
            data_dict["displacement"] = data_dict["displacement"][shuffle_index]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][shuffle_index]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][shuffle_index]
        if "organ_segment" in data_dict.keys():
            data_dict["organ_segment"] = data_dict["organ_segment"][shuffle_index]
        if "organ_instance" in data_dict.keys():
            data_dict["organ_instance"] = data_dict["organ_instance"][shuffle_index]
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][mask]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][mask]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][mask]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][mask]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][mask]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][mask]
        if "organ_segment" in data_dict.keys():
            data_dict["organ_segment"] = data_dict["organ_segment"][mask]
        if "organ_instance" in data_dict.keys():
            data_dict["organ_instance"] = data_dict["organ_instance"][mask]
        return data_dict


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
            self,
            view_keys=("coord", "color", "normal", "origin_coord"),
            view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"].copy()
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        # data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict



@TRANSFORMS.register_module()
class InstanceParserMAX(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"].copy()
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_max
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        # data_dict["instance"] = instance
        stem_mask = segment == 0
        centroid[stem_mask] = coord[stem_mask].min(0)
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


@TRANSFORMS.register_module()
class InstanceParserV3(object):
    def __init__(self, segment_ignore_index=[-1], instance_ignore_index=-1,
                 organ_segment_ignore_index=[-1]):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.organ_segment_ignore_index = organ_segment_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]

        organ_segment = data_dict["organ_segment"]
        organ_instance = data_dict["organ_instance"]

        instance, centroid, bottom, bbox, organ_instance, organ_centroid, organ_bottom, organ_bbox = self._get_combined_info(
            coord=coord.copy(), segment=segment, instance=instance, organ_segment=organ_segment,
            organ_instance=organ_instance)

        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["instance_bottom"] = bottom
        data_dict["bbox"] = bbox

        data_dict["organ_instance"] = organ_instance
        data_dict["organ_instance_centroid"] = organ_centroid
        data_dict["organ_instance_bottom"] = organ_bottom
        data_dict["organ_bbox"] = organ_bbox
        return data_dict

    def _get_combined_info(self, coord, segment, instance, organ_segment, organ_instance):
        # Preprocessing for instance-level information
        mask = ~np.in1d(segment, self.segment_ignore_index)
        instance[~mask] = self.instance_ignore_index

        unique_instances, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique_instances)
        instance[mask] = inverse

        # Initialize storage for instance-level information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bottom = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index

        organ_mask = ~np.in1d(organ_segment, self.organ_segment_ignore_index)
        organ_instance[~organ_mask] = self.instance_ignore_index
        # Initialize storage for organ-level information
        organ_unique_instances, organ_inverse = np.unique(organ_instance[organ_mask], return_inverse=True)
        organ_instance_num = len(organ_unique_instances)
        organ_instance[organ_mask] = organ_inverse

        organ_centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        organ_bottom = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        organ_bbox = np.ones((organ_instance_num, 8)) * self.instance_ignore_index

        for inst_id in unique_instances:
            if inst_id == self.instance_ignore_index:
                continue  # Skip ignored instances

            inst_mask = instance == inst_id
            inst_coords = coord[inst_mask]

            if inst_coords.size == 0:
                continue

            inst_segment = segment[inst_mask]

            # Instance-level bbox and centroid
            bbox_min = inst_coords.min(axis=0)
            bbox_max = inst_coords.max(axis=0)
            inst_centroid = inst_coords.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=inst_coords.dtype)
            bbox_class = np.array([inst_segment[0]], dtype=inst_coords.dtype)
            centroid[inst_mask] = inst_centroid
            bbox[inst_id] = np.concatenate([bbox_center, bbox_size, bbox_theta, bbox_class])

            # Processing for organ-level information
            inst_organ_segment = organ_segment[inst_mask]
            inst_organ_instance = organ_instance[inst_mask]
            stem_coords = inst_coords[inst_organ_segment == 0]
            if len(stem_coords) == 0:
                continue
            stem_tree = cKDTree(stem_coords)

            stem_bottom = stem_coords[stem_coords[:, 2].argmin()]
            stem_instance_id = np.unique(inst_organ_instance[inst_organ_segment == 0])
            organ_bottom[organ_instance == stem_instance_id] = stem_bottom
            bottom[organ_instance == stem_instance_id] = stem_bottom

            stem_centroid = stem_coords.mean(0)
            organ_centroid[organ_instance == stem_instance_id] = stem_centroid

            leaf_instance_ids = np.unique(inst_organ_instance[inst_organ_segment == 1])

            for leaf_id in leaf_instance_ids:
                if leaf_id == self.instance_ignore_index:
                    continue

                leaf_mask = inst_organ_instance == leaf_id
                leaf_coords = inst_coords[leaf_mask]

                leaf_inst_mask = organ_instance == leaf_id
                # Calculate nearest leaf point to stem as organ_centroid
                if len(stem_coords) > 0 and len(leaf_coords) > 0:
                    # distances = np.linalg.norm(leaf_coords[:, np.newaxis, :] - stem_coords[np.newaxis, :, :], axis=2)
                    # closest_leaf_idx = np.argmin(np.min(distances, axis=1))
                    # organ_centroid[leaf_inst_mask] = leaf_coords[closest_leaf_idx]
                    # 查询每个叶子点到所有茎点的最近距离和对应的索引
                    distances, indices = stem_tree.query(leaf_coords)
                    closest_leaf_idx = np.argmin(distances)
                    organ_bottom[leaf_inst_mask] = leaf_coords[closest_leaf_idx]

                # Calculate organ_instance bbox
                organ_min = leaf_coords.min(axis=0)
                organ_max = leaf_coords.max(axis=0)
                leaf_centroid = leaf_coords.mean(0)
                organ_centroid[leaf_inst_mask] = leaf_centroid
                organ_center = (organ_min + organ_max) / 2
                organ_size = organ_max - organ_min
                organ_bbox[leaf_id] = np.concatenate(
                    [organ_center, organ_size, np.zeros(1), np.array([inst_organ_segment[leaf_mask][0]])])

        return instance, centroid, bottom, bbox, organ_instance, organ_centroid, organ_bottom, organ_bbox


@TRANSFORMS.register_module()
class InstanceParserV1(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            # bbox_centroid = coord_.mean(0)
            bbox_centroid = coord_[coord_[:, 2].argmin()]
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


@TRANSFORMS.register_module()
class InstanceParserV4(object):
    def __init__(self, segment_ignore_index=[-1], instance_ignore_index=-1, ):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]

        organ_segment = data_dict["organ_segment"]
        organ_instance = data_dict["organ_instance"]

        instance, centroid, bottom, stem_v = self._get_single_info(
            coord=coord.copy(), segment=segment, instance=instance, organ_segment=organ_segment,
            organ_instance=organ_instance)

        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["instance_bottom"] = bottom
        data_dict["instance_stem_v"] = stem_v

        return data_dict

    def _get_single_info(self, coord, segment, instance, organ_segment, organ_instance):
        # Preprocessing for instance-level information
        mask = ~np.in1d(segment, self.segment_ignore_index)
        instance[~mask] = self.instance_ignore_index

        unique_instances, inverse = np.unique(instance[mask], return_inverse=True)
        instance[mask] = inverse

        # Initialize storage for instance-level information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bottom = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        stem_v = np.zeros((coord.shape[0], 3))

        for inst_id in unique_instances:
            if inst_id == self.instance_ignore_index:
                continue  # Skip ignored instances

            inst_mask = instance == inst_id
            inst_coords = coord[inst_mask]

            if inst_coords.size == 0:
                continue

            # Instance-level bbox and centroid
            inst_centroid = inst_coords.mean(0)
            centroid[inst_mask] = inst_centroid

            # Processing for organ-level information
            inst_organ_segment = organ_segment[inst_mask]
            inst_organ_instance = organ_instance[inst_mask]
            stem_coords = inst_coords[inst_organ_segment == 0]
            # if len(stem_coords) == 0:
            #     stem_bottom = inst_coords[inst_coords[:, 2].argmin()]
            #     pca = PCA(n_components=3)
            #     pca.fit(inst_coords)
            #     stem_growth_direction = pca.components_[0]
                # stem_instance_id = np.unique(inst_organ_instance[inst_organ_segment == 0])
            try:
                stem_bottom = stem_coords[stem_coords[:, 2].argmin()]
                pca = PCA(n_components=3)
                pca.fit(stem_coords)
                stem_growth_direction = pca.components_[0]
            except:
                stem_bottom = inst_coords[inst_coords[:, 2].argmin()]
                pca = PCA(n_components=3)
                pca.fit(inst_coords)
                stem_growth_direction = pca.components_[0]
            bottom[inst_mask] = stem_bottom
            stem_v[inst_mask] = stem_growth_direction

        return instance, centroid, bottom, stem_v


@TRANSFORMS.register_module()
class InstanceParserV41(object):
    def __init__(self, segment_ignore_index=[-1], instance_ignore_index=-1, ):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]

        organ_segment = data_dict["organ_segment"]
        organ_instance = data_dict["organ_instance"]

        instance, centroid, bottom, stem_v = self._get_single_info(
            coord=coord.copy(), segment=segment, instance=instance, organ_segment=organ_segment,
            organ_instance=organ_instance)

        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["instance_bottom"] = bottom
        data_dict["instance_stem_v"] = stem_v

        return data_dict

    def _get_single_info(self, coord, segment, instance, organ_segment, organ_instance):
        # Preprocessing for instance-level information
        mask = ~np.in1d(segment, self.segment_ignore_index)
        instance[~mask] = self.instance_ignore_index

        unique_instances, inverse = np.unique(instance[mask], return_inverse=True)
        instance[mask] = inverse

        # Initialize storage for instance-level information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bottom = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        stem_v = np.zeros((coord.shape[0], 3))

        for inst_id in unique_instances:
            if inst_id == self.instance_ignore_index:
                continue  # Skip ignored instances

            inst_mask = instance == inst_id
            inst_coords = coord[inst_mask]

            if inst_coords.size == 0:
                continue

            # Instance-level bbox and centroid
            inst_centroid = inst_coords.mean(0)
            centroid[inst_mask] = inst_centroid

            # Processing for organ-level information
            inst_organ_segment = organ_segment[inst_mask]
            inst_organ_instance = organ_instance[inst_mask]
            stem_coords = inst_coords[inst_organ_segment == 0]
            # if len(stem_coords) == 0:
            #     stem_bottom = inst_coords[inst_coords[:, 2].argmin()]
            #     pca = PCA(n_components=3)
            #     pca.fit(inst_coords)
            #     stem_growth_direction = pca.components_[0]
                # stem_instance_id = np.unique(inst_organ_instance[inst_organ_segment == 0])
            try:
                stem_bottom = stem_coords[stem_coords[:, 2].argmin()]
                pca = PCA(n_components=3)
                pca.fit(stem_coords)
                stem_growth_direction = pca.components_[0]
            except:
                stem_bottom = inst_coords[inst_coords[:, 2].argmin()]
                stem_growth_direction = [0,0,1]
            bottom[inst_mask] = stem_bottom
            stem_v[inst_mask] = stem_growth_direction

        return instance, centroid, bottom, stem_v

@TRANSFORMS.register_module()
class InstanceParserV44(object):
    def __init__(self, segment_ignore_index=[-1], instance_ignore_index=-1, ):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]

        instance, centroid, bottom, stem_v = self._get_single_info(
            coord=coord.copy(), segment=segment, instance=instance)

        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["instance_bottom"] = bottom
        data_dict["instance_stem_v"] = stem_v

        return data_dict

    def _get_single_info(self, coord, segment, instance):
        # Preprocessing for instance-level information
        mask = ~np.in1d(segment, self.segment_ignore_index)
        instance[~mask] = self.instance_ignore_index

        unique_instances, inverse = np.unique(instance[mask], return_inverse=True)
        instance[mask] = inverse

        # Initialize storage for instance-level information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bottom = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        stem_v = np.zeros((coord.shape[0], 3))

        for inst_id in unique_instances:
            if inst_id == self.instance_ignore_index:
                continue  # Skip ignored instances

            inst_mask = instance == inst_id
            inst_coords = coord[inst_mask]

            if inst_coords.size == 0:
                continue

            # Instance-level bbox and centroid
            inst_centroid = inst_coords.mean(0)
            centroid[inst_mask] = inst_centroid

            # Calculate bottom and stem growth direction from all instance points
            try:
                inst_bottom = inst_coords[inst_coords[:, 2].argmin()]
                pca = PCA(n_components=3)
                pca.fit(inst_coords)
                stem_growth_direction = pca.components_[0]
            except:
                inst_bottom = inst_coords[inst_coords[:, 2].argmin()]
                stem_growth_direction = [0,0,1]

            if stem_growth_direction[2] < 0:
                stem_growth_direction = -stem_growth_direction  # Ensure the growth direction points upward
            bottom[inst_mask] = inst_bottom
            stem_v[inst_mask] = stem_growth_direction

        return instance, centroid, bottom, stem_v






@TRANSFORMS.register_module()
class InstanceParserV6(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)

        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index
        # plant
        # INST_ID = 1
        # mask_ = instance == INST_ID
        # coord_ = coord[mask_]
        # bbox_centroid = coord_.mean(0)
        # bbox_centroid[0] = 0
        # bbox_centroid[1] = 0
        # centroid[mask_] = bbox_centroid
        # # ground
        # INST_ID = 0
        # mask_ = instance == INST_ID
        # centroid[mask_] = -bbox_centroid

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_centroid[0] = 0
            bbox_centroid[1] = 0
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict



@TRANSFORMS.register_module()
class InstanceParserV5(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1, instance_segment_ignore_index=(-1, 0)):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.instance_segment_ignore_index = instance_segment_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        mask = ~np.in1d(segment, self.instance_segment_ignore_index)
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


@TRANSFORMS.register_module()
class InstanceChange(object):
    def __call__(self, data_dict):
        # 将 organ instance 变为 instance，原来的 instance 变为 single_instance
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        instance_centroid = data_dict["instance_centroid"]
        instance_bottom = data_dict["instance_bottom"]

        organ_segment = data_dict["organ_segment"]
        organ_instance = data_dict["organ_instance"]
        organ_instance_centroid = data_dict["organ_instance_centroid"]
        organ_instance_bottom = data_dict["organ_instance_bottom"]

        data_dict["segment"] = organ_segment
        data_dict["instance"] = organ_instance
        data_dict["instance_centroid"] = organ_instance_centroid
        data_dict["instance_bottom"] = organ_instance_bottom

        data_dict["single_segment"] = segment
        data_dict["single_instance"] = instance
        data_dict["single_instance_centroid"] = instance_centroid
        data_dict["single_instance_bottom"] = instance_bottom
        return data_dict


@TRANSFORMS.register_module()
class SimpleInstanceChange(object):
    def __call__(self, data_dict):
        # 将 organ instance 变为 instance，原来的 instance 变为 single_instance
        segment = data_dict["segment"]
        instance = data_dict["instance"]

        organ_segment = data_dict["organ_segment"]
        organ_instance = data_dict["organ_instance"]

        data_dict["segment"] = organ_segment
        data_dict["instance"] = organ_instance

        data_dict["single_segment"] = segment
        data_dict["single_instance"] = instance

        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
