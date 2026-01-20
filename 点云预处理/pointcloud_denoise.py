#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于距离聚类的点云去噪模块 (Distance-based Clustering Denoising)

核心算法：
1. DBSCAN密度聚类 - 识别任意形状簇，最大簇保留策略
2. 统计滤波 - 基于邻域均值距离偏差
3. 半径滤波 - 基于邻域点数阈值

适用场景：轨枕检测等结构化点云的噪声剔除
"""

import os
import logging
import numpy as np
import open3d as o3d
from datetime import datetime
from typing import Tuple, Optional, Union, List


# ============================================================================
# 日志配置
# ============================================================================

_logger: Optional[logging.Logger] = None


def setup_logger(
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    console_output: bool = False
) -> logging.Logger:
    """配置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pointcloud_denoise_{timestamp}.log")
    
    logger = logging.getLogger("PointCloudDenoise")
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    return logger


def get_logger() -> logging.Logger:
    """获取全局日志器"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def set_logger(logger: logging.Logger):
    """设置自定义日志器"""
    global _logger
    _logger = logger


# ============================================================================
# DBSCAN密度聚类去噪（核心方法）
# ============================================================================

def dbscan_denoise(
    points: Union[np.ndarray, o3d.geometry.PointCloud],
    eps: float = 0.05,
    min_points: int = 10,
    keep_largest: bool = True,
    min_cluster_size: Optional[int] = None
) -> Tuple[o3d.geometry.PointCloud, dict]:
    """
    基于DBSCAN密度聚类的点云去噪
    
    算法原理：
    - 核心点条件：|N_ε(p)| ≥ MinPts
    - 噪声点标记为-1
    - 最大簇保留策略：选取点数最多的簇作为主体
    
    Args:
        points: 输入点云（numpy数组或Open3D点云）
        eps: 邻域搜索半径ε
        min_points: 最小邻域点数MinPts
        keep_largest: 是否仅保留最大簇
        min_cluster_size: 有效簇的最小点数，None则使用min_points
    
    Returns:
        (denoised_pcd, info): 去噪点云和处理信息
    """
    logger = get_logger()
    
    # 输入转换
    if isinstance(points, np.ndarray):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("点云数组必须是n×3格式")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd = points
    
    n_original = len(pcd.points)
    
    # 空点云检查
    if n_original == 0:
        logger.warning("输入点云为空")
        return pcd, {'original_points': 0, 'denoised_points': 0}
    
    logger.info(f"DBSCAN去噪开始: 点数={n_original}, eps={eps}, min_points={min_points}")
    
    # 执行DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    # 统计聚类结果
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels[unique_labels >= 0])
    n_noise = np.sum(labels == -1)
    
    logger.info(f"聚类结果: {n_clusters} 个簇, {n_noise} 个噪声点")
    
    # 记录各簇信息
    cluster_info = []
    for label, count in zip(unique_labels, counts):
        if label >= 0:
            cluster_info.append((label, count))
            logger.debug(f"  簇{label}: {count} 点")
    
    if not cluster_info:
        logger.warning("未检测到有效簇，返回原始点云")
        return pcd, {
            'original_points': n_original,
            'denoised_points': n_original,
            'n_clusters': 0,
            'noise_points': n_noise,
            'kept_clusters': [],
        }
    
    # 确定保留的簇
    if min_cluster_size is None:
        min_cluster_size = min_points
    
    if keep_largest:
        # 最大簇保留策略
        largest_cluster = max(cluster_info, key=lambda x: x[1])
        kept_labels = [largest_cluster[0]]
        logger.info(f"最大簇保留: 簇{largest_cluster[0]}, {largest_cluster[1]} 点")
    else:
        # 保留所有大于阈值的簇
        kept_labels = [label for label, count in cluster_info if count >= min_cluster_size]
        logger.info(f"保留 {len(kept_labels)} 个簇 (size >= {min_cluster_size})")
    
    # 提取保留的点
    mask = np.isin(labels, kept_labels)
    indices = np.where(mask)[0]
    pcd_denoised = pcd.select_by_index(indices)
    
    n_denoised = len(pcd_denoised.points)
    noise_ratio = (n_original - n_denoised) / n_original * 100
    
    logger.info(f"去噪完成: {n_original} -> {n_denoised} (去除{noise_ratio:.1f}%)")
    
    info = {
        'original_points': n_original,
        'denoised_points': n_denoised,
        'noise_removed': n_original - n_denoised,
        'noise_ratio': noise_ratio,
        'n_clusters': n_clusters,
        'noise_points': n_noise,
        'kept_clusters': kept_labels,
        'cluster_sizes': dict(cluster_info),
        'eps': eps,
        'min_points': min_points,
    }
    
    return pcd_denoised, info


# ============================================================================
# 统计滤波去噪
# ============================================================================

def statistical_denoise(
    points: Union[np.ndarray, o3d.geometry.PointCloud],
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> Tuple[o3d.geometry.PointCloud, dict]:
    """
    统计滤波去噪
    
    原理：计算每个点到其k近邻的平均距离，剔除距离偏差超过阈值的点
    条件：|p̄_i - μ| > α·σ 则为噪声
    
    Args:
        points: 输入点云
        nb_neighbors: 近邻点数k
        std_ratio: 标准差倍数α
    
    Returns:
        (denoised_pcd, info): 去噪点云和处理信息
    """
    logger = get_logger()
    
    # 输入转换
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd = points
    
    n_original = len(pcd.points)
    logger.info(f"统计滤波开始: 点数={n_original}, k={nb_neighbors}, α={std_ratio}")
    
    # 执行统计滤波
    pcd_denoised, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    n_denoised = len(pcd_denoised.points)
    noise_ratio = (n_original - n_denoised) / n_original * 100
    
    logger.info(f"统计滤波完成: {n_original} -> {n_denoised} (去除{noise_ratio:.1f}%)")
    
    info = {
        'original_points': n_original,
        'denoised_points': n_denoised,
        'noise_removed': n_original - n_denoised,
        'noise_ratio': noise_ratio,
        'nb_neighbors': nb_neighbors,
        'std_ratio': std_ratio,
    }
    
    return pcd_denoised, info


# ============================================================================
# 半径滤波去噪
# ============================================================================

def radius_denoise(
    points: Union[np.ndarray, o3d.geometry.PointCloud],
    radius: float = 0.05,
    min_neighbors: int = 5
) -> Tuple[o3d.geometry.PointCloud, dict]:
    """
    半径滤波去噪
    
    原理：统计每个点在半径r内的邻居数，数量不足则剔除
    条件：|{q : ||q - p|| < r}| < MinPts 则为噪声
    
    Args:
        points: 输入点云
        radius: 搜索半径r
        min_neighbors: 最小邻居数MinPts
    
    Returns:
        (denoised_pcd, info): 去噪点云和处理信息
    """
    logger = get_logger()
    
    # 输入转换
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd = points
    
    n_original = len(pcd.points)
    logger.info(f"半径滤波开始: 点数={n_original}, r={radius}, min_neighbors={min_neighbors}")
    
    # 执行半径滤波
    pcd_denoised, inlier_indices = pcd.remove_radius_outlier(
        nb_points=min_neighbors,
        radius=radius
    )
    
    n_denoised = len(pcd_denoised.points)
    noise_ratio = (n_original - n_denoised) / n_original * 100
    
    logger.info(f"半径滤波完成: {n_original} -> {n_denoised} (去除{noise_ratio:.1f}%)")
    
    info = {
        'original_points': n_original,
        'denoised_points': n_denoised,
        'noise_removed': n_original - n_denoised,
        'noise_ratio': noise_ratio,
        'radius': radius,
        'min_neighbors': min_neighbors,
    }
    
    return pcd_denoised, info


# ============================================================================
# 组合去噪（流水线）
# ============================================================================

def pipeline_denoise(
    points: Union[np.ndarray, o3d.geometry.PointCloud],
    methods: List[str] = ['statistical', 'dbscan'],
    statistical_params: dict = None,
    radius_params: dict = None,
    dbscan_params: dict = None
) -> Tuple[o3d.geometry.PointCloud, dict]:
    """
    组合去噪流水线
    
    Args:
        points: 输入点云
        methods: 去噪方法顺序，可选 'statistical', 'radius', 'dbscan'
        statistical_params: 统计滤波参数
        radius_params: 半径滤波参数
        dbscan_params: DBSCAN参数
    
    Returns:
        (denoised_pcd, info): 去噪点云和处理信息
    """
    logger = get_logger()
    
    # 默认参数
    if statistical_params is None:
        statistical_params = {'nb_neighbors': 20, 'std_ratio': 2.0}
    if radius_params is None:
        radius_params = {'radius': 0.05, 'min_neighbors': 5}
    if dbscan_params is None:
        dbscan_params = {'eps': 0.05, 'min_points': 10, 'keep_largest': True}
    
    # 输入转换
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd = points
    
    n_original = len(pcd.points)
    logger.info(f"组合去噪开始: 点数={n_original}, 流程={methods}")
    
    pipeline_info = {
        'original_points': n_original,
        'methods': methods,
        'stages': [],
    }
    
    current_pcd = pcd
    
    for method in methods:
        if method == 'statistical':
            current_pcd, info = statistical_denoise(current_pcd, **statistical_params)
        elif method == 'radius':
            current_pcd, info = radius_denoise(current_pcd, **radius_params)
        elif method == 'dbscan':
            current_pcd, info = dbscan_denoise(current_pcd, **dbscan_params)
        else:
            logger.warning(f"未知方法: {method}，跳过")
            continue
        
        pipeline_info['stages'].append({
            'method': method,
            'points_after': info['denoised_points'],
            'removed': info['noise_removed'],
        })
    
    n_final = len(current_pcd.points)
    total_removed = n_original - n_final
    
    pipeline_info['denoised_points'] = n_final
    pipeline_info['total_removed'] = total_removed
    pipeline_info['total_noise_ratio'] = total_removed / n_original * 100 if n_original > 0 else 0
    
    logger.info(f"组合去噪完成: {n_original} -> {n_final} (总去除{pipeline_info['total_noise_ratio']:.1f}%)")
    
    return current_pcd, pipeline_info


# ============================================================================
# 自适应参数估计
# ============================================================================

def estimate_dbscan_params(
    points: Union[np.ndarray, o3d.geometry.PointCloud],
    k: int = 10,
    percentile: float = 95
) -> Tuple[float, int]:
    """
    自适应估计DBSCAN参数
    
    原理：基于k近邻距离分布估计eps
    
    Args:
        points: 输入点云
        k: 近邻数
        percentile: 使用的距离百分位数
    
    Returns:
        (eps, min_points): 建议的参数
    """
    logger = get_logger()
    
    if isinstance(points, o3d.geometry.PointCloud):
        points = np.asarray(points.points)
    
    from scipy.spatial import KDTree
    
    kdtree = KDTree(points)
    distances, _ = kdtree.query(points, k=k+1)
    
    # 使用第k近邻距离（排除自身）
    knn_distances = distances[:, -1]
    
    # 使用百分位数作为eps
    eps = np.percentile(knn_distances, percentile)
    min_points = k
    
    logger.info(f"自适应参数估计: eps={eps:.6f}, min_points={min_points}")
    logger.debug(f"k近邻距离: mean={np.mean(knn_distances):.6f}, std={np.std(knn_distances):.6f}")
    
    return eps, min_points


# ============================================================================
# 批量去噪
# ============================================================================

def batch_denoise(
    pointclouds: List[Union[np.ndarray, o3d.geometry.PointCloud]],
    method: str = 'dbscan',
    **kwargs
) -> List[Tuple[o3d.geometry.PointCloud, dict]]:
    """
    批量点云去噪
    
    Args:
        pointclouds: 点云列表
        method: 去噪方法 ('dbscan', 'statistical', 'radius', 'pipeline')
        **kwargs: 方法参数
    
    Returns:
        [(denoised_pcd, info), ...] 结果列表
    """
    logger = get_logger()
    
    n = len(pointclouds)
    print(f"批量去噪: {n} 个点云, 方法={method}")
    logger.info(f"批量去噪开始: {n} 个点云, method={method}")
    
    # 方法映射
    method_funcs = {
        'dbscan': dbscan_denoise,
        'statistical': statistical_denoise,
        'radius': radius_denoise,
        'pipeline': pipeline_denoise,
    }
    
    if method not in method_funcs:
        raise ValueError(f"未知方法: {method}, 支持: {list(method_funcs.keys())}")
    
    denoise_func = method_funcs[method]
    results = []
    
    total_original = 0
    total_denoised = 0
    
    for i, pcd in enumerate(pointclouds):
        logger.info(f"[{i+1}/{n}] 处理中...")
        
        try:
            pcd_denoised, info = denoise_func(pcd, **kwargs)
            results.append((pcd_denoised, info))
            
            total_original += info['original_points']
            total_denoised += info['denoised_points']
            
        except Exception as e:
            logger.error(f"[{i+1}/{n}] 失败: {e}")
            results.append((None, {'error': str(e)}))
    
    if total_original > 0:
        overall_ratio = (total_original - total_denoised) / total_original * 100
        print(f"批量去噪完成: {total_original} -> {total_denoised} (去除{overall_ratio:.1f}%)")
        logger.info(f"批量完成: {total_original} -> {total_denoised} ({overall_ratio:.1f}%去除)")
    
    return results


# ============================================================================
# 去噪质量评估
# ============================================================================

def evaluate_denoise_quality(
    original: Union[np.ndarray, o3d.geometry.PointCloud],
    denoised: Union[np.ndarray, o3d.geometry.PointCloud]
) -> dict:
    """
    评估去噪质量
    
    Args:
        original: 原始点云
        denoised: 去噪点云
    
    Returns:
        质量指标字典
    """
    logger = get_logger()
    
    if isinstance(original, o3d.geometry.PointCloud):
        original = np.asarray(original.points)
    if isinstance(denoised, o3d.geometry.PointCloud):
        denoised = np.asarray(denoised.points)
    
    n_original = len(original)
    n_denoised = len(denoised)
    
    # 计算包围盒变化
    bbox_original = np.max(original, axis=0) - np.min(original, axis=0)
    bbox_denoised = np.max(denoised, axis=0) - np.min(denoised, axis=0)
    
    # 体积比
    vol_original = np.prod(bbox_original)
    vol_denoised = np.prod(bbox_denoised)
    vol_ratio = vol_denoised / vol_original if vol_original > 0 else 0
    
    # 密度变化
    density_original = n_original / vol_original if vol_original > 0 else 0
    density_denoised = n_denoised / vol_denoised if vol_denoised > 0 else 0
    
    metrics = {
        'original_points': n_original,
        'denoised_points': n_denoised,
        'removal_ratio': (n_original - n_denoised) / n_original if n_original > 0 else 0,
        'bbox_original': bbox_original.tolist(),
        'bbox_denoised': bbox_denoised.tolist(),
        'volume_ratio': vol_ratio,
        'density_original': density_original,
        'density_denoised': density_denoised,
    }
    
    logger.info(f"去噪评估: 去除率={metrics['removal_ratio']*100:.1f}%, 体积保留={vol_ratio*100:.1f}%")
    
    return metrics


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    logger = setup_logger(log_dir="./logs", console_output=False)
    
    # 创建测试点云（含噪声）
    np.random.seed(42)
    
    # 主体点云（模拟轨枕）
    n_main = 10000
    main_points = np.random.rand(n_main, 3) * np.array([10, 2, 1])  # 长条形
    
    # 添加噪声点
    n_noise = 500
    noise_points = np.random.rand(n_noise, 3) * np.array([15, 5, 3]) - np.array([2, 1, 1])
    
    # 合并
    all_points = np.vstack([main_points, noise_points])
    
    # 自适应参数估计
    eps, min_pts = estimate_dbscan_params(all_points, k=10)
    
    # DBSCAN去噪
    pcd_denoised, info = dbscan_denoise(
        all_points,
        eps=eps,
        min_points=min_pts,
        keep_largest=True
    )
    
    print(f"DBSCAN去噪: {info['original_points']} -> {info['denoised_points']}")
    
    # 组合去噪示例
    pcd_pipeline, pipe_info = pipeline_denoise(
        all_points,
        methods=['statistical', 'dbscan'],
        statistical_params={'nb_neighbors': 20, 'std_ratio': 2.0},
        dbscan_params={'eps': 0.1, 'min_points': 10, 'keep_largest': True}
    )
    
    print(f"组合去噪: {pipe_info['original_points']} -> {pipe_info['denoised_points']}")
    print("详情见 ./logs/")
