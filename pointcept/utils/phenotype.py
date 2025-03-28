# -*- coding:utf-8 -*-
"""
coding   : utf-8
@Project ：Pointcept-1.5.1 
@File    ：phenotype.py
@IDE     ：PyCharm 
@Author  ：yangxin6
@Date    ：2025/1/11 上午10:27 
"""
import numpy as np
from sklearn.decomposition import PCA

def cal_feat_v(xyz):
    pca = PCA(n_components=3)
    pca.fit(xyz)
    v1, v2, v3 = pca.components_
    return v1, v2, v3

def cal_h_and_crown_width(xyz, v1, v2, v3):
    # 使用矩阵乘法一次计算所有主成分的投影
    projections = np.dot(xyz, np.array([v1, v2, v3]).T)
    proj_v1, proj_v2, proj_v3 = projections[:, 0], projections[:, 1], projections[:, 2]

    # 株高：v1方向上的最大值与最小值之差
    height = abs(proj_v1.max() - proj_v1.min())

    # 冠幅：v2和v3方向上的最大值与最小值之差的乘积
    width_v2 = abs(proj_v2.max() - proj_v2.min())
    width_v3 = abs(proj_v3.max() - proj_v3.min())
    # crown_width = width_v2 * width_v3
    width_v2 = max([width_v3, width_v2])
    return height, width_v2, width_v3

def cal_azimuth(xyz):
    projected_points = xyz[:, :2]  # 只取x, y坐标
    pca = PCA(n_components=1)
    pca.fit(projected_points)
    main_direction = pca.components_[0]
    center_point = np.mean(projected_points, axis=0)
    azimuth = np.arctan2(main_direction[1], main_direction[0])  # 与X的夹角
    # np.arctan2(main_direction[0], main_direction[1])  与Y的夹角
    azimuth_deg = np.degrees(azimuth)
    return azimuth_deg, main_direction, center_point

def cal_one_phenotype(xyz, l, points_th):
    unique_l = np.unique(l)
    results = {}
    count = 0
    for i in unique_l:
        if i == -1:
            continue
        i_mask = l == i
        i_xyz = xyz[i_mask]
        i_xyz -= np.mean(i_xyz, axis=0)
        # 对这个数据进行去噪
        i_xyz_denoised = i_xyz
        # if len(i_xyz) < 40:
        #     i_xyz_denoised = i_xyz
        # else:
        #     i_xyz_denoised = denoise_point_cloud(i_xyz, k=32, alpha=5)
        # show_2pcd(i_xyz, i_xyz_denoised)
        if points_th[0] < len(i_xyz_denoised) < points_th[1]:
            v1, v2, v3 = cal_feat_v(i_xyz_denoised)
            height, width_v2, width_v3 = cal_h_and_crown_width(i_xyz_denoised, v1, v2, v3)
            if height < 0 or width_v2 < 0:
                continue
            # 计算方位角
            i_azimuth_deg, i_main_direction, i_center_point = cal_azimuth(i_xyz_denoised)
            results[i] = [height, width_v2, width_v3, i_azimuth_deg]
            count += 1
        else:
            results[i] = [-1, -1, -1, -1]
            continue

    return results


def calculate_ape(gt, pred):
    """
    计算绝对百分比误差 (APE)

    参数:
    gt (float): 真实值
    pred (float): 预测值

    返回:
    float: APE值
    """
    if gt == -1 or pred == -1:
        return -1  # 避免除以零的情况
    return abs((gt - pred) / gt)

from scipy.stats import entropy

def calculate_js_divergence(real_values, pred_values, bins=50):
    """
    计算真实玉米群体和预测玉米群体在高度或宽度上的 JS Divergence。

    Args:
        real_values (array-like): 真实群体的某个表型参数（如高度或宽度）。
        pred_values (array-like): 预测群体的某个表型参数（如高度或宽度）。
        bins (int): 计算概率分布时的分箱数量。

    Returns:
        float: JS Divergence 值。
    """
    # 计算直方图 (概率分布)
    real_hist, bin_edges = np.histogram(real_values, bins=bins, density=True)
    pred_hist, _ = np.histogram(pred_values, bins=bin_edges, density=True)

    # 平滑化，避免零概率问题
    real_hist += 1e-9
    pred_hist += 1e-9

    # 归一化为概率分布
    real_hist /= real_hist.sum()
    pred_hist /= pred_hist.sum()

    # 计算中间分布
    m = 0.5 * (real_hist + pred_hist)

    # 计算 JS Divergence
    js_div = 0.5 * (entropy(real_hist, m) + entropy(pred_hist, m))
    return js_div