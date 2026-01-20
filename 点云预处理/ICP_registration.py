#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于ICP的点云配准模块 (ICP-based Point Cloud Registration)

核心算法：Iterative Closest Point (Besl & McKay, 1992)
流程：质心对齐 → 初始变换 → KD-tree最近邻搜索 → SVD求解刚体变换 → 迭代收敛

功能：
- 单对点云配准
- 多点云批量配准（与模板一一对应）
- 支持自定义初始变换矩阵
- 重叠区域提取（Z轴阈值分割）
"""

import os
import logging
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Optional, List, Union
from datetime import datetime


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
    log_file = os.path.join(log_dir, f"icp_registration_{timestamp}.log")
    
    logger = logging.getLogger("ICPRegistration")
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
# 核心ICP算法
# ============================================================================

def icp_registration(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: Optional[np.ndarray] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    distance_threshold: Optional[float] = None,
    align_centroid: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    ICP点云配准
    
    算法流程：
    1. 质心对齐（可选）
    2. 应用初始变换
    3. KD-tree寻找对应点
    4. SVD估计最优刚体变换
    5. 迭代直至收敛
    
    Args:
        source: 源点云 n×3
        target: 目标点云 m×3
        init_transform: 初始4×4变换矩阵，None则使用单位矩阵
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值（变换变化量）
        distance_threshold: 对应点距离阈值，超过则剔除
        align_centroid: 是否先进行质心对齐
    
    Returns:
        (final_transform, aligned_source, info): 最终变换矩阵、配准后点云、配准信息
    """
    logger = get_logger()
    
    # 输入验证
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("源点云必须是n×3数组")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("目标点云必须是m×3数组")
    
    logger.info(f"ICP配准开始: 源点云={len(source)}点, 目标点云={len(target)}点")
    logger.info(f"参数: max_iter={max_iterations}, tol={tolerance}, align_centroid={align_centroid}")
    
    # 复制源点云用于变换
    src = source.copy()
    
    # -------------------------
    # Step 1: 质心对齐（预处理）
    # -------------------------
    centroid_transform = np.eye(4)
    if align_centroid:
        centroid_transform = _compute_centroid_alignment(src, target)
        src = _apply_transform(src, centroid_transform)
        logger.info(f"质心对齐完成: 平移向量={centroid_transform[:3, 3]}")
    
    # -------------------------
    # Step 2: 应用初始变换
    # -------------------------
    if init_transform is not None:
        if init_transform.shape != (4, 4):
            raise ValueError("初始变换矩阵必须是4×4")
        src = _apply_transform(src, init_transform)
        logger.info("已应用自定义初始变换矩阵")
    else:
        init_transform = np.eye(4)
    
    # 构建目标点云KD-tree
    kdtree = KDTree(target)
    
    # 累积变换矩阵
    cumulative_transform = np.eye(4)
    
    # -------------------------
    # Step 3-5: 迭代优化
    # -------------------------
    prev_error = np.inf
    converged = False
    
    for iteration in range(max_iterations):
        # (1) 寻找对应点
        distances, indices = kdtree.query(src)
        
        # 距离阈值过滤
        if distance_threshold is not None:
            valid_mask = distances < distance_threshold
            if np.sum(valid_mask) < 10:
                logger.warning(f"迭代{iteration}: 有效对应点过少({np.sum(valid_mask)})")
                break
            src_matched = src[valid_mask]
            tgt_matched = target[indices[valid_mask]]
        else:
            src_matched = src
            tgt_matched = target[indices]
        
        # 计算当前误差
        mean_error = np.mean(distances)
        
        logger.debug(f"迭代{iteration}: 对应点={len(src_matched)}, 平均误差={mean_error:.6f}")
        
        # (2) SVD估计最优变换
        R, t = _estimate_rigid_transform(src_matched, tgt_matched)
        
        # 构建4×4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        # (3) 应用变换
        src = _apply_transform(src, transform)
        cumulative_transform = transform @ cumulative_transform
        
        # (4) 收敛判断
        error_change = abs(prev_error - mean_error)
        
        if error_change < tolerance:
            logger.info(f"迭代{iteration}: 收敛(误差变化={error_change:.2e} < {tolerance})")
            converged = True
            break
        
        # 检查变换变化量
        transform_change = np.linalg.norm(transform - np.eye(4))
        if transform_change < tolerance:
            logger.info(f"迭代{iteration}: 收敛(变换变化={transform_change:.2e})")
            converged = True
            break
        
        prev_error = mean_error
    
    if not converged:
        logger.warning(f"达到最大迭代次数({max_iterations})，未完全收敛")
    
    # 组合最终变换：初始变换 × 质心对齐 × ICP迭代变换
    final_transform = cumulative_transform @ init_transform @ centroid_transform
    
    # 计算最终误差
    aligned_source = _apply_transform(source, final_transform)
    final_distances, _ = kdtree.query(aligned_source)
    final_error = np.mean(final_distances)
    
    logger.info(f"ICP完成: 最终误差={final_error:.6f}, 迭代次数={iteration+1}")
    
    # 配准信息
    info = {
        'iterations': iteration + 1,
        'converged': converged,
        'final_error': final_error,
        'init_error': np.mean(KDTree(target).query(source)[0]),
    }
    
    return final_transform, aligned_source, info


def _compute_centroid_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """计算质心对齐变换矩阵"""
    src_centroid = np.mean(source, axis=0)
    tgt_centroid = np.mean(target, axis=0)
    
    translation = tgt_centroid - src_centroid
    
    transform = np.eye(4)
    transform[:3, 3] = translation
    
    return transform


def _estimate_rigid_transform(
    source: np.ndarray,
    target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SVD求解最优刚体变换
    
    最小化: E = Σ||R·pi + t - qi||²
    
    Returns:
        (R, t): 3×3旋转矩阵, 3×1平移向量
    """
    # 去质心
    src_centroid = np.mean(source, axis=0)
    tgt_centroid = np.mean(target, axis=0)
    
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    
    # 协方差矩阵
    H = src_centered.T @ tgt_centered
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 旋转矩阵
    R = Vt.T @ U.T
    
    # 处理反射情况（确保det(R)=1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 平移向量
    t = tgt_centroid - R @ src_centroid
    
    return R, t


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """应用4×4变换矩阵到点云"""
    ones = np.ones((len(points), 1))
    points_homo = np.hstack([points, ones])
    transformed = (transform @ points_homo.T).T
    return transformed[:, :3]


# ============================================================================
# 重叠区域提取
# ============================================================================

def extract_overlap_region(
    points: np.ndarray,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    percentile_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    提取点云重叠区域（基于Z轴阈值分割）
    
    Args:
        points: n×3点云
        z_min, z_max: Z轴范围，None则不限制
        percentile_range: 使用百分位数确定范围，如(80, 100)表示Z值在80%-100%范围
    
    Returns:
        提取的点云子集
    """
    logger = get_logger()
    
    z_coords = points[:, 2]
    
    # 使用百分位数确定范围
    if percentile_range is not None:
        z_min = np.percentile(z_coords, percentile_range[0])
        z_max = np.percentile(z_coords, percentile_range[1])
        logger.debug(f"百分位范围 {percentile_range} -> Z:[{z_min:.4f}, {z_max:.4f}]")
    
    # 创建掩码
    mask = np.ones(len(points), dtype=bool)
    if z_min is not None:
        mask &= z_coords >= z_min
    if z_max is not None:
        mask &= z_coords <= z_max
    
    extracted = points[mask]
    logger.info(f"重叠区域提取: {len(points)} -> {len(extracted)} 点")
    
    return extracted


# ============================================================================
# 批量配准
# ============================================================================

def batch_registration(
    sources: List[np.ndarray],
    targets: List[np.ndarray],
    init_transforms: Optional[List[np.ndarray]] = None,
    extract_overlap: bool = False,
    overlap_percentile: Tuple[float, float] = (80, 100),
    **icp_kwargs
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """
    批量点云配准（源与目标一一对应）
    
    Args:
        sources: 源点云列表
        targets: 目标点云列表（模板）
        init_transforms: 初始变换矩阵列表，None则全部使用单位矩阵
        extract_overlap: 是否先提取重叠区域再配准
        overlap_percentile: 重叠区域Z轴百分位范围
        **icp_kwargs: 传递给icp_registration的参数
    
    Returns:
        [(transform, aligned, info), ...] 配准结果列表
    """
    logger = get_logger()
    
    if len(sources) != len(targets):
        raise ValueError(f"源点云数量({len(sources)})与目标数量({len(targets)})不匹配")
    
    n = len(sources)
    
    if init_transforms is None:
        init_transforms = [None] * n
    elif len(init_transforms) != n:
        raise ValueError(f"初始变换数量({len(init_transforms)})与点云数量({n})不匹配")
    
    print(f"批量配准: {n} 对点云")
    logger.info(f"批量配准开始: {n} 对点云, extract_overlap={extract_overlap}")
    
    results = []
    success_count = 0
    
    for i in range(n):
        logger.info(f"[{i+1}/{n}] 配准第 {i+1} 对点云")
        
        try:
            src = sources[i]
            tgt = targets[i]
            
            # 提取重叠区域
            if extract_overlap:
                src = extract_overlap_region(src, percentile_range=overlap_percentile)
                tgt = extract_overlap_region(tgt, percentile_range=overlap_percentile)
            
            # ICP配准
            transform, aligned, info = icp_registration(
                src, tgt,
                init_transform=init_transforms[i],
                **icp_kwargs
            )
            
            # 将变换应用到原始完整点云
            full_aligned = _apply_transform(sources[i], transform)
            
            results.append((transform, full_aligned, info))
            success_count += 1
            
            logger.info(f"[{i+1}/{n}] 成功: 误差={info['final_error']:.6f}")
            
        except Exception as e:
            logger.error(f"[{i+1}/{n}] 失败: {e}")
            results.append((None, None, {'error': str(e)}))
    
    print(f"批量配准完成: {success_count}/{n} 成功")
    logger.info(f"批量配准完成: 成功={success_count}/{n}")
    
    return results


# ============================================================================
# 配准质量评估
# ============================================================================

def evaluate_registration(
    aligned: np.ndarray,
    target: np.ndarray,
    distance_threshold: float = 0.1
) -> dict:
    """
    评估配准质量
    
    Args:
        aligned: 配准后的点云
        target: 目标点云
        distance_threshold: 内点距离阈值
    
    Returns:
        评估指标字典
    """
    logger = get_logger()
    
    kdtree = KDTree(target)
    distances, _ = kdtree.query(aligned)
    
    # 内点比例
    inlier_mask = distances < distance_threshold
    inlier_ratio = np.mean(inlier_mask)
    
    metrics = {
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'max_distance': float(np.max(distances)),
        'std_distance': float(np.std(distances)),
        'rmse': float(np.sqrt(np.mean(distances ** 2))),
        'inlier_ratio': float(inlier_ratio),
        'inlier_count': int(np.sum(inlier_mask)),
    }
    
    logger.info(f"配准评估: RMSE={metrics['rmse']:.6f}, 内点率={inlier_ratio*100:.1f}%")
    
    return metrics


# ============================================================================
# 变换矩阵工具
# ============================================================================

def create_transform_matrix(
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    rotation_euler: Optional[Tuple[float, float, float]] = None,
    rotation_axis_angle: Optional[Tuple[np.ndarray, float]] = None
) -> np.ndarray:
    """
    创建4×4变换矩阵
    
    Args:
        rotation: 3×3旋转矩阵
        translation: 3×1平移向量
        rotation_euler: 欧拉角(rx, ry, rz)，单位弧度
        rotation_axis_angle: (轴向量, 角度)
    
    Returns:
        4×4变换矩阵
    """
    transform = np.eye(4)
    
    # 旋转矩阵
    if rotation is not None:
        transform[:3, :3] = rotation
    elif rotation_euler is not None:
        transform[:3, :3] = _euler_to_rotation(*rotation_euler)
    elif rotation_axis_angle is not None:
        axis, angle = rotation_axis_angle
        transform[:3, :3] = _axis_angle_to_rotation(axis, angle)
    
    # 平移向量
    if translation is not None:
        transform[:3, 3] = translation
    
    return transform


def _euler_to_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """欧拉角转旋转矩阵（ZYX顺序）"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx


def _axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """轴角转旋转矩阵（Rodrigues公式）"""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def decompose_transform(transform: np.ndarray) -> dict:
    """分解变换矩阵为旋转和平移"""
    logger = get_logger()
    
    R = transform[:3, :3]
    t = transform[:3, 3]
    
    # 提取欧拉角
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0
    
    result = {
        'rotation_matrix': R,
        'translation': t,
        'euler_angles': np.array([rx, ry, rz]),
        'euler_degrees': np.degrees([rx, ry, rz]),
    }
    
    logger.debug(f"变换分解: 平移={t}, 欧拉角(度)={result['euler_degrees']}")
    
    return result


# ============================================================================
# 合并配准后点云
# ============================================================================

def merge_registered_pointclouds(
    pointclouds: List[np.ndarray],
    transforms: List[np.ndarray],
    voxel_size: Optional[float] = None
) -> np.ndarray:
    """
    合并配准后的点云
    
    Args:
        pointclouds: 原始点云列表
        transforms: 对应变换矩阵列表
        voxel_size: 体素降采样大小，None则不降采样
    
    Returns:
        合并后的点云
    """
    logger = get_logger()
    
    all_points = []
    
    for i, (pcd, T) in enumerate(zip(pointclouds, transforms)):
        if T is None:
            logger.warning(f"点云{i}无变换矩阵，跳过")
            continue
        transformed = _apply_transform(pcd, T)
        all_points.append(transformed)
    
    if not all_points:
        return np.array([])
    
    merged = np.vstack(all_points)
    logger.info(f"合并点云: {len(pointclouds)} 个 -> {len(merged)} 点")
    
    # 体素降采样
    if voxel_size is not None:
        merged = _voxel_downsample(merged, voxel_size)
        logger.info(f"降采样后: {len(merged)} 点")
    
    return merged


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """简单体素降采样"""
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return points[unique_indices]


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    logger = setup_logger(log_dir="./logs", console_output=False)
    
    # 创建测试数据
    np.random.seed(42)
    
    # 目标点云（模板）
    target = np.random.rand(1000, 3) * 10
    
    # 源点云（带已知变换）
    true_R = _euler_to_rotation(0.1, 0.2, 0.15)
    true_t = np.array([1.0, 2.0, 0.5])
    true_transform = create_transform_matrix(rotation=true_R, translation=true_t)
    
    source = _apply_transform(target, np.linalg.inv(true_transform))
    source += np.random.randn(*source.shape) * 0.01  # 添加噪声
    
    # 单对配准
    transform, aligned, info = icp_registration(
        source, target,
        align_centroid=True,
        max_iterations=50,
        tolerance=1e-6
    )
    
    # 评估
    metrics = evaluate_registration(aligned, target)
    
    # 分解变换
    decomposed = decompose_transform(transform)
    
    print(f"配准完成: RMSE={metrics['rmse']:.6f}")
    print(f"恢复平移: {decomposed['translation']}")
    print(f"真实平移: {true_t}")
    print("详情见 ./logs/")
