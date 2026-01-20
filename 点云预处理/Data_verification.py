#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云数据验证与校准模块

本模块用于验证点云是否在预定包围盒内，并在必要时进行强制校准。
校准流程包括：核密度估计几何中心、平移到包围盒中心、按包围盒剪裁。

"""

import os
import logging
import numpy as np
from typing import Tuple, Optional, List, Union
from scipy.stats import gaussian_kde


# ============================================================================
# 日志配置
# ============================================================================
_logger: Optional[logging.Logger] = None


def setup_logger(
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    log_name: str = "data_verification"
) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        log_dir: 日志文件保存目录
        log_level: 日志级别
        log_name: 日志名称
        
    Returns:
        配置好的日志记录器
    """
    global _logger
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    
    # 如果日志记录器已经有处理器，先清空
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{log_name}.log"),
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    获取全局日志器
    
    Returns:
        全局日志记录器
    """
    global _logger
    
    if _logger is None:
        setup_logger()
    
    return _logger


def set_logger(logger: logging.Logger) -> None:
    """
    设置自定义日志器
    
    Args:
        logger: 自定义日志器
    """
    global _logger
    _logger = logger


# ============================================================================
# 点云包围盒验证
# ============================================================================

def verify_pointcloud_in_bbox(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """
    验证点云是否在预定的包围盒内
    
    Args:
        points: N×3的点云数据
        bbox_min: 包围盒最小值 [x_min, y_min, z_min]
        bbox_max: 包围盒最大值 [x_max, y_max, z_max]
        
    Returns:
        (is_valid, outlier_mask)
        is_valid: 点云是否全部在包围盒内
        outlier_mask: 离群点掩码，True表示在包围盒外
    """
    logger = get_logger()
    
    # 检查点云是否在包围盒内
    within_x = np.logical_and(points[:, 0] >= bbox_min[0], points[:, 0] <= bbox_max[0])
    within_y = np.logical_and(points[:, 1] >= bbox_min[1], points[:, 1] <= bbox_max[1])
    within_z = np.logical_and(points[:, 2] >= bbox_min[2], points[:, 2] <= bbox_max[2])
    
    in_bbox_mask = np.logical_and(np.logical_and(within_x, within_y), within_z)
    outlier_mask = np.logical_not(in_bbox_mask)
    
    is_valid = np.all(in_bbox_mask)
    
    logger.info(f"点云验证结果：{is_valid}")
    logger.info(f"总点数：{points.shape[0]}")
    logger.info(f"在包围盒内的点数：{np.sum(in_bbox_mask)}")
    logger.info(f"离群点数：{np.sum(outlier_mask)}")
    
    if not is_valid:
        print(f"点云验证失败：{np.sum(outlier_mask)}个点在包围盒外")
    
    return is_valid, outlier_mask


def calculate_bbox(
    points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算点云的包围盒
    
    Args:
        points: N×3的点云数据
        
    Returns:
        (bbox_min, bbox_max, bbox_center)
        bbox_min: 包围盒最小值
        bbox_max: 包围盒最大值
        bbox_center: 包围盒中心
    """
    logger = get_logger()
    
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    
    logger.info(f"点云包围盒计算完成")
    logger.info(f"包围盒最小值：{bbox_min}")
    logger.info(f"包围盒最大值：{bbox_max}")
    logger.info(f"包围盒中心：{bbox_center}")
    
    return bbox_min, bbox_max, bbox_center


# ============================================================================
# 核密度估计几何中心
# ============================================================================

def estimate_geometric_center(
    points: np.ndarray,
    kde_bandwidth: float = 0.1
) -> np.ndarray:
    """
    使用核密度估计计算点云的几何中心
    
    Args:
        points: N×3的点云数据
        kde_bandwidth: 核密度估计的带宽参数
        
    Returns:
        geometric_center: 估计的几何中心
    """
    logger = get_logger()
    
    # 计算每个维度的核密度估计
    x_kde = gaussian_kde(points[:, 0], bw_method=kde_bandwidth)
    y_kde = gaussian_kde(points[:, 1], bw_method=kde_bandwidth)
    z_kde = gaussian_kde(points[:, 2], bw_method=kde_bandwidth)
    
    # 在每个维度上找到密度最大的点
    x_values = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 1000)
    y_values = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 1000)
    z_values = np.linspace(np.min(points[:, 2]), np.max(points[:, 2]), 1000)
    
    x_density = x_kde(x_values)
    y_density = y_kde(y_values)
    z_density = z_kde(z_values)
    
    x_center = x_values[np.argmax(x_density)]
    y_center = y_values[np.argmax(y_density)]
    z_center = z_values[np.argmax(z_density)]
    
    geometric_center = np.array([x_center, y_center, z_center])
    
    logger.info(f"核密度估计几何中心：{geometric_center}")
    
    return geometric_center


def estimate_geometric_center_3d(
    points: np.ndarray,
    kde_bandwidth: float = 0.1
) -> np.ndarray:
    """
    使用3D核密度估计计算点云的几何中心
    
    Args:
        points: N×3的点云数据
        kde_bandwidth: 核密度估计的带宽参数
        
    Returns:
        geometric_center: 估计的几何中心
    """
    logger = get_logger()
    
    # 3D核密度估计
    kde = gaussian_kde(points.T, bw_method=kde_bandwidth)
    
    # 创建网格用于计算密度
    x_range = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 50)
    y_range = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 50)
    z_range = np.linspace(np.min(points[:, 2]), np.max(points[:, 2]), 50)
    
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # 计算密度值
    density = kde(positions)
    
    # 找到密度最大的位置
    max_density_index = np.argmax(density)
    geometric_center = positions[:, max_density_index]
    
    logger.info(f"3D核密度估计几何中心：{geometric_center}")
    
    return geometric_center


# ============================================================================
# 点云强制校准
# ============================================================================

def calibrate_pointcloud(
    points: np.ndarray,
    target_bbox_min: np.ndarray,
    target_bbox_max: np.ndarray,
    use_3d_kde: bool = False
) -> np.ndarray:
    """
    强制校准点云到目标包围盒
    
    Args:
        points: N×3的点云数据
        target_bbox_min: 目标包围盒最小值
        target_bbox_max: 目标包围盒最大值
        use_3d_kde: 是否使用3D核密度估计
        
    Returns:
        calibrated_points: 校准后的点云数据
    """
    logger = get_logger()
    
    print(f"开始校准点云，总点数：{points.shape[0]}")
    
    # 步骤1：计算目标包围盒中心
    target_bbox_center = (target_bbox_min + target_bbox_max) / 2
    logger.info(f"目标包围盒中心：{target_bbox_center}")
    
    # 步骤2：估计点云几何中心
    if use_3d_kde:
        geometric_center = estimate_geometric_center_3d(points)
    else:
        geometric_center = estimate_geometric_center(points)
    
    # 步骤3：平移点云，使几何中心与目标包围盒中心重合
    translation_vector = target_bbox_center - geometric_center
    translated_points = points + translation_vector
    
    logger.info(f"平移向量：{translation_vector}")
    logger.info(f"平移后点云范围：{np.min(translated_points, axis=0)} ~ {np.max(translated_points, axis=0)}")
    
    # 步骤4：按目标包围盒剪裁点云
    mask_x = np.logical_and(translated_points[:, 0] >= target_bbox_min[0], translated_points[:, 0] <= target_bbox_max[0])
    mask_y = np.logical_and(translated_points[:, 1] >= target_bbox_min[1], translated_points[:, 1] <= target_bbox_max[1])
    mask_z = np.logical_and(translated_points[:, 2] >= target_bbox_min[2], translated_points[:, 2] <= target_bbox_max[2])
    
    clip_mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    calibrated_points = translated_points[clip_mask]
    
    logger.info(f"剪裁后点云点数：{calibrated_points.shape[0]}")
    logger.info(f"剪裁掉的点数：{points.shape[0] - calibrated_points.shape[0]}")
    
    print(f"点云校准完成：{points.shape[0]} → {calibrated_points.shape[0]}个点")
    
    return calibrated_points


def verify_and_calibrate(
    points: np.ndarray,
    target_bbox_min: np.ndarray,
    target_bbox_max: np.ndarray,
    auto_calibrate: bool = True,
    use_3d_kde: bool = False
) -> Tuple[np.ndarray, bool]:
    """
    验证点云并在必要时进行校准
    
    Args:
        points: N×3的点云数据
        target_bbox_min: 目标包围盒最小值
        target_bbox_max: 目标包围盒最大值
        auto_calibrate: 是否自动校准
        use_3d_kde: 是否使用3D核密度估计
        
    Returns:
        (result_points, was_calibrated)
        result_points: 验证或校准后的点云
        was_calibrated: 是否进行了校准
    """
    logger = get_logger()
    
    # 验证点云是否在包围盒内
    is_valid, outlier_mask = verify_pointcloud_in_bbox(points, target_bbox_min, target_bbox_max)
    
    if is_valid:
        logger.info("点云验证通过，无需校准")
        return points, False
    
    if not auto_calibrate:
        logger.warning("点云验证失败，且未启用自动校准")
        return points, False
    
    # 进行强制校准
    logger.info("点云验证失败，开始强制校准")
    calibrated_points = calibrate_pointcloud(points, target_bbox_min, target_bbox_max, use_3d_kde)
    
    return calibrated_points, True


# ============================================================================
# 主函数（示例）
# ============================================================================
if __name__ == "__main__":
    # 设置日志
    setup_logger(log_level=logging.DEBUG)
    logger = get_logger()
    
    # 生成测试点云
    np.random.seed(42)
    test_points = np.random.rand(10000, 3) * 2.0 - 1.0  # 范围在[-1, 1]的随机点云
    
    # 设置目标包围盒
    target_bbox_min = np.array([-0.8, -0.8, -0.8])
    target_bbox_max = np.array([0.8, 0.8, 0.8])
    
    logger.info("开始点云验证与校准测试")
    logger.info(f"测试点云范围：{np.min(test_points, axis=0)} ~ {np.max(test_points, axis=0)}")
    logger.info(f"目标包围盒：{target_bbox_min} ~ {target_bbox_max}")
    
    # 验证并校准点云
    result_points, was_calibrated = verify_and_calibrate(test_points, target_bbox_min, target_bbox_max)
    
    logger.info(f"测试完成，是否校准：{was_calibrated}")
    logger.info(f"结果点云范围：{np.min(result_points, axis=0)} ~ {np.max(result_points, axis=0)}")
    logger.info(f"结果点云点数：{result_points.shape[0]}")
