#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于自适应策略的点云降采样模块 (Adaptive Point Cloud Downsampling)

核心算法：
1. 计算点云包围盒对角线长度作为尺度参数
2. 根据点数规模动态选取系数α，计算体素尺寸 voxel_size = α × diagonal
3. 最小点数约束保护，不足则缩小体素重试

适用场景：轨枕检测等精度敏感型点云应用
"""

import os
import logging
import numpy as np
import open3d as o3d
from datetime import datetime
from typing import Tuple, Optional, Union


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
    log_file = os.path.join(log_dir, f"adaptive_downsample_{timestamp}.log")
    
    logger = logging.getLogger("AdaptiveDownsample")
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
# 包围盒与尺度计算
# ============================================================================

def compute_bounding_box(points: np.ndarray) -> dict:
    """
    计算点云包围盒信息
    
    Args:
        points: n×3点云数组
    
    Returns:
        包围盒信息字典
    """
    logger = get_logger()
    
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    # 各轴尺寸
    extent = max_bound - min_bound
    
    # 对角线长度：diagonal = sqrt(dx² + dy² + dz²)
    diagonal = np.linalg.norm(extent)
    
    bbox_info = {
        'min_bound': min_bound,
        'max_bound': max_bound,
        'extent': extent,
        'diagonal': diagonal,
        'volume': np.prod(extent),
        'center': (min_bound + max_bound) / 2,
    }
    
    logger.debug(f"包围盒: extent={extent}, diagonal={diagonal:.4f}")
    
    return bbox_info


def estimate_point_density(points: np.ndarray, bbox_info: dict) -> float:
    """
    估计点云密度（点数/体积）
    
    Args:
        points: 点云数组
        bbox_info: 包围盒信息
    
    Returns:
        点密度
    """
    volume = bbox_info['volume']
    if volume < 1e-10:
        return 0.0
    return len(points) / volume


# ============================================================================
# 自适应体素尺寸计算
# ============================================================================

def compute_adaptive_voxel_size(
    n_points: int,
    diagonal: float,
    alpha_large: float = 0.0005,
    alpha_medium: float = 0.002,
    alpha_small: float = 0.005,
    threshold_large: int = 100000,
    threshold_medium: int = 50000
) -> Tuple[float, float]:
    """
    根据点数规模自适应计算体素尺寸
    
    公式：voxel_size = α × diagonal
    
    分段策略：
    - n > 100000: α = 0.0005 (大规模，保留细节)
    - 50000 < n ≤ 100000: α = 0.002 (中等规模)
    - n ≤ 50000: α = 0.005 (小规模，降低冗余)
    
    Args:
        n_points: 点云点数
        diagonal: 包围盒对角线长度
        alpha_large/medium/small: 各规模对应的系数
        threshold_large/medium: 规模阈值
    
    Returns:
        (voxel_size, alpha): 体素尺寸和使用的系数
    """
    logger = get_logger()
    
    if n_points > threshold_large:
        alpha = alpha_large
        scale = "large"
    elif n_points > threshold_medium:
        alpha = alpha_medium
        scale = "medium"
    else:
        alpha = alpha_small
        scale = "small"
    
    voxel_size = alpha * diagonal
    
    logger.info(f"自适应体素: n={n_points}, scale={scale}, α={alpha}, voxel_size={voxel_size:.6f}")
    
    return voxel_size, alpha


# ============================================================================
# 核心降采样函数
# ============================================================================

def adaptive_voxel_downsample(
    points: Union[np.ndarray, o3d.geometry.PointCloud],
    min_points: int = 50000,
    voxel_size: Optional[float] = None,
    alpha_large: float = 0.0005,
    alpha_medium: float = 0.002,
    alpha_small: float = 0.005,
    max_retry: int = 3,
    retry_factor: float = 0.7
) -> Tuple[o3d.geometry.PointCloud, dict]:
    """
    自适应体素降采样
    
    算法流程：
    1. 计算包围盒对角线长度
    2. 根据点数规模选取系数α，计算voxel_size = α × diagonal
    3. 执行体素降采样
    4. 若结果点数 < min_points，缩小体素尺寸重试
    5. 重试仍不足则返回原始点云
    
    Args:
        points: 输入点云（numpy数组或Open3D点云）
        min_points: 最小点数约束（默认50000）
        voxel_size: 手动指定体素尺寸，None则自适应计算
        alpha_large/medium/small: 分段系数
        max_retry: 最大重试次数
        retry_factor: 重试时体素缩小系数
    
    Returns:
        (downsampled_pcd, info): 降采样点云和处理信息
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
    
    original_points = np.asarray(pcd.points)
    n_original = len(original_points)
    
    logger.info(f"开始自适应降采样: 原始点数={n_original}")
    
    # 计算包围盒
    bbox_info = compute_bounding_box(original_points)
    diagonal = bbox_info['diagonal']
    
    logger.info(f"包围盒对角线: {diagonal:.4f}")
    
    # 计算自适应体素尺寸
    if voxel_size is None:
        voxel_size, alpha = compute_adaptive_voxel_size(
            n_original, diagonal,
            alpha_large, alpha_medium, alpha_small
        )
    else:
        alpha = voxel_size / diagonal if diagonal > 0 else 0
        logger.info(f"使用指定体素尺寸: {voxel_size:.6f}")
    
    # 迭代降采样（带最小点数保护）
    current_voxel_size = voxel_size
    
    for attempt in range(max_retry + 1):
        # 执行降采样
        pcd_down = pcd.voxel_down_sample(voxel_size=current_voxel_size)
        n_down = len(pcd_down.points)
        
        ratio = n_down / n_original * 100
        logger.info(f"尝试{attempt+1}: voxel_size={current_voxel_size:.6f}, 结果点数={n_down} ({ratio:.1f}%)")
        
        # 检查最小点数约束
        if n_down >= min_points:
            logger.info(f"降采样成功: {n_original} -> {n_down}")
            break
        
        # 原始点数本身不足，直接返回
        if n_original < min_points:
            logger.warning(f"原始点数({n_original})已低于最小约束({min_points})，返回原始点云")
            pcd_down = pcd
            n_down = n_original
            break
        
        # 缩小体素尺寸重试
        if attempt < max_retry:
            current_voxel_size *= retry_factor
            logger.info(f"点数不足，缩小体素尺寸重试: {current_voxel_size:.6f}")
        else:
            # 达到最大重试次数，返回原始点云
            logger.warning(f"达到最大重试次数，返回原始点云")
            pcd_down = pcd
            n_down = n_original
    
    # 处理信息
    info = {
        'original_points': n_original,
        'downsampled_points': len(pcd_down.points),
        'reduction_ratio': 1 - len(pcd_down.points) / n_original,
        'voxel_size': current_voxel_size,
        'alpha': alpha,
        'diagonal': diagonal,
        'bbox': bbox_info,
        'attempts': attempt + 1,
    }
    
    return pcd_down, info


# ============================================================================
# 批量降采样
# ============================================================================

def batch_adaptive_downsample(
    pointclouds: list,
    min_points: int = 50000,
    **kwargs
) -> list:
    """
    批量自适应降采样
    
    Args:
        pointclouds: 点云列表（numpy数组或Open3D点云）
        min_points: 最小点数约束
        **kwargs: 传递给adaptive_voxel_downsample的参数
    
    Returns:
        [(downsampled_pcd, info), ...] 结果列表
    """
    logger = get_logger()
    
    n = len(pointclouds)
    print(f"批量降采样: {n} 个点云")
    logger.info(f"批量降采样开始: {n} 个点云, min_points={min_points}")
    
    results = []
    total_original = 0
    total_down = 0
    
    for i, pcd in enumerate(pointclouds):
        logger.info(f"[{i+1}/{n}] 处理中...")
        
        try:
            pcd_down, info = adaptive_voxel_downsample(pcd, min_points, **kwargs)
            results.append((pcd_down, info))
            
            total_original += info['original_points']
            total_down += info['downsampled_points']
            
        except Exception as e:
            logger.error(f"[{i+1}/{n}] 失败: {e}")
            results.append((None, {'error': str(e)}))
    
    # 统计
    if total_original > 0:
        overall_ratio = (1 - total_down / total_original) * 100
        print(f"批量降采样完成: 总点数 {total_original} -> {total_down} (减少{overall_ratio:.1f}%)")
        logger.info(f"批量完成: {total_original} -> {total_down} ({overall_ratio:.1f}% 减少)")
    
    return results


# ============================================================================
# 其他降采样方法（补充）
# ============================================================================

def uniform_downsample(
    pcd: o3d.geometry.PointCloud,
    every_k_points: int = 10
) -> o3d.geometry.PointCloud:
    """均匀降采样（每隔k个点取一个）"""
    logger = get_logger()
    
    n_original = len(pcd.points)
    pcd_down = pcd.uniform_down_sample(every_k_points=every_k_points)
    n_down = len(pcd_down.points)
    
    logger.info(f"均匀降采样: k={every_k_points}, {n_original} -> {n_down}")
    
    return pcd_down


def random_downsample(
    pcd: o3d.geometry.PointCloud,
    ratio: float = 0.5
) -> o3d.geometry.PointCloud:
    """随机降采样"""
    logger = get_logger()
    
    n_original = len(pcd.points)
    n_target = int(n_original * ratio)
    
    indices = np.random.choice(n_original, size=n_target, replace=False)
    pcd_down = pcd.select_by_index(indices)
    
    logger.info(f"随机降采样: ratio={ratio}, {n_original} -> {len(pcd_down.points)}")
    
    return pcd_down


def farthest_point_downsample(
    points: np.ndarray,
    n_samples: int
) -> np.ndarray:
    """
    最远点采样（FPS）- 保持几何分布均匀
    
    Args:
        points: n×3点云数组
        n_samples: 采样点数
    
    Returns:
        采样后的点云数组
    """
    logger = get_logger()
    
    n = len(points)
    if n_samples >= n:
        return points.copy()
    
    # 初始化
    sampled_indices = np.zeros(n_samples, dtype=np.int32)
    distances = np.full(n, np.inf)
    
    # 随机选择第一个点
    sampled_indices[0] = np.random.randint(n)
    
    for i in range(1, n_samples):
        # 更新距离（到已采样点集的最小距离）
        last_point = points[sampled_indices[i-1]]
        dist_to_last = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dist_to_last)
        
        # 选择距离最大的点
        sampled_indices[i] = np.argmax(distances)
    
    result = points[sampled_indices]
    logger.info(f"FPS降采样: {n} -> {n_samples}")
    
    return result


# ============================================================================
# 降采样质量评估
# ============================================================================

def evaluate_downsample_quality(
    original: np.ndarray,
    downsampled: np.ndarray,
    n_samples: int = 10000
) -> dict:
    """
    评估降采样质量
    
    指标：
    - 覆盖率：原始点到最近降采样点的平均距离
    - 分布均匀性：降采样点的最近邻距离标准差
    
    Args:
        original: 原始点云
        downsampled: 降采样点云
        n_samples: 评估采样数（原始点云过大时）
    
    Returns:
        质量指标字典
    """
    logger = get_logger()
    
    from scipy.spatial import KDTree
    
    # 采样评估（原始点云过大时）
    if len(original) > n_samples:
        indices = np.random.choice(len(original), n_samples, replace=False)
        original_sample = original[indices]
    else:
        original_sample = original
    
    # 构建降采样点云KD-tree
    kdtree = KDTree(downsampled)
    
    # 覆盖率：原始点到降采样点的距离
    distances, _ = kdtree.query(original_sample)
    coverage_mean = np.mean(distances)
    coverage_max = np.max(distances)
    
    # 分布均匀性：降采样点的最近邻距离
    if len(downsampled) > 1:
        kdtree_down = KDTree(downsampled)
        nn_distances, _ = kdtree_down.query(downsampled, k=2)
        nn_distances = nn_distances[:, 1]  # 排除自身
        uniformity_std = np.std(nn_distances)
        uniformity_mean = np.mean(nn_distances)
    else:
        uniformity_std = 0
        uniformity_mean = 0
    
    metrics = {
        'coverage_mean': float(coverage_mean),
        'coverage_max': float(coverage_max),
        'uniformity_mean': float(uniformity_mean),
        'uniformity_std': float(uniformity_std),
        'compression_ratio': len(downsampled) / len(original),
    }
    
    logger.info(f"降采样质量: coverage_mean={coverage_mean:.6f}, uniformity_std={uniformity_std:.6f}")
    
    return metrics


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    logger = setup_logger(log_dir="./logs", console_output=False)
    
    # 创建测试点云（模拟大规模点云）
    np.random.seed(42)
    
    # 120000点的测试数据
    n_test = 120000
    points = np.random.rand(n_test, 3) * 100  # 100×100×100的空间
    
    # 自适应降采样
    pcd_down, info = adaptive_voxel_downsample(
        points,
        min_points=50000,
        alpha_large=0.0005,
        alpha_medium=0.002,
        alpha_small=0.005
    )
    
    print(f"降采样完成: {info['original_points']} -> {info['downsampled_points']}")
    print(f"体素尺寸: {info['voxel_size']:.6f}, 系数α: {info['alpha']}")
    
    # 质量评估
    metrics = evaluate_downsample_quality(points, np.asarray(pcd_down.points))
    print(f"覆盖率(平均距离): {metrics['coverage_mean']:.6f}")
    
    print("详情见 ./logs/")
