#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云平面提取模块 (Point Cloud Plane Extraction)

核心功能：
1. 基于比例参考点和法向量约束的平面提取
2. 支持RANSAC和区域增长两种平面拟合方法
3. 返回原始点索引，避免拟合误差引入系统偏差

使用场景：工业场景下的构件尺寸测量、平面分割等
"""

import os
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, List, Union
from scipy.spatial import KDTree


# ============================================================================# 日志配置
# ============================================================================_
_logger: Optional[logging.Logger] = None

def setup_logger(
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    console_output: bool = False
) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        log_dir: 日志文件目录
        log_level: 日志级别
        console_output: 是否输出到控制台
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"plane_extraction_{timestamp}.log")
    
    logger = logging.getLogger("PlaneExtractor")
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


# ============================================================================# 辅助函数
# ============================================================================_
def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算点云包围盒
    
    Args:
        points: N×3点云数组
    
    Returns:
        (min_bound, max_bound): 包围盒的最小和最大边界
    """
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound

def get_reference_point(points: np.ndarray, ratio_point: np.ndarray) -> np.ndarray:
    """
    根据比例参考点获取实际空间位置
    
    Args:
        points: N×3点云数组
        ratio_point: 比例参考点 [rx, ry, rz]
    
    Returns:
        实际空间位置 p0
    """
    min_bound, max_bound = compute_bounding_box(points)
    p0 = min_bound + ratio_point * (max_bound - min_bound)
    return p0

def get_local_region(points: np.ndarray, reference_point: np.ndarray, radius_ratio: float = 0.1) -> np.ndarray:
    """
    获取局部搜索区域
    
    Args:
        points: N×3点云数组
        reference_point: 参考点 p0
        radius_ratio: 搜索半径与包围盒对角线长度的比例
    
    Returns:
        局部搜索区域内的点索引
    """
    min_bound, max_bound = compute_bounding_box(points)
    diagonal = np.linalg.norm(max_bound - min_bound)
    radius = radius_ratio * diagonal
    
    # 计算每个点到参考点的距离
    distances = np.linalg.norm(points - reference_point, axis=1)
    
    # 返回在搜索半径内的点索引
    return np.where(distances <= radius)[0]

def compute_plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    从三个点计算平面方程
    
    Args:
        p1, p2, p3: 三个不共线的点
    
    Returns:
        (normal, d): 法向量和平面方程参数
    """
    # 计算法向量
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    
    # 检查是否共线
    if np.linalg.norm(normal) < 1e-8:
        raise ValueError("三点共线，无法确定平面")
    
    # 归一化法向量
    normal = normal / np.linalg.norm(normal)
    
    # 计算平面方程参数 d
    d = -np.dot(normal, p1)
    
    return normal, d

def get_point_plane_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """
    计算点到平面的距离
    
    Args:
        points: N×3点云数组
        normal: 平面法向量
        d: 平面方程参数
    
    Returns:
        每个点到平面的距离
    """
    return np.abs(np.dot(points, normal) + d)

def check_normal_consistency(normal1: np.ndarray, normal2: np.ndarray, angle_threshold: float = 15.0) -> bool:
    """
    检查两个法向量是否一致（夹角小于阈值）
    
    Args:
        normal1, normal2: 两个法向量
        angle_threshold: 夹角阈值（度）
    
    Returns:
        是否一致
    """
    # 计算夹角余弦值
    cos_angle = np.dot(normal1, normal2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
    
    # 转换为角度
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle <= angle_threshold


# ============================================================================# RANSAC平面提取
# ============================================================================_
def ransac_plane_extraction(
    points: np.ndarray,
    reference_point: np.ndarray,
    target_normal: np.ndarray,
    local_region: np.ndarray,
    distance_threshold: float = 0.01,
    max_iterations: int = 1000,
    confidence: float = 0.99
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    使用RANSAC方法提取平面
    
    Args:
        points: N×3点云数组
        reference_point: 参考点 p0
        target_normal: 目标法向量
        local_region: 局部搜索区域内的点索引
        distance_threshold: 点到平面的距离阈值
        max_iterations: 最大迭代次数
        confidence: 置信度
    
    Returns:
        (best_inliers, best_normal, best_d): 最佳内点索引、最佳法向量、最佳平面参数
    """
    logger = get_logger()
    logger.info("开始RANSAC平面提取")
    
    # 获取局部区域内的点
    local_points = points[local_region]
    
    best_inliers = np.array([])
    best_normal = np.zeros(3)
    best_d = 0.0
    best_inlier_count = 0
    
    # 计算所需迭代次数
    # 假设内点比例为50%，需要多少次迭代才能有confidence的概率至少有一次采样全是内点
    iterations = max_iterations
    
    for i in range(iterations):
        # 随机选择3个点
        sample_indices = np.random.choice(len(local_region), 3, replace=False)
        sample_points = local_points[sample_indices]
        
        try:
            # 计算平面方程
            normal, d = compute_plane_from_points(sample_points[0], sample_points[1], sample_points[2])
        except ValueError:
            # 三点共线，跳过
            continue
        
        # 检查法向量是否与目标法向量一致
        if not check_normal_consistency(normal, target_normal):
            continue
        
        # 计算所有点到平面的距离
        distances = get_point_plane_distance(local_points, normal, d)
        
        # 统计内点
        inliers_local = np.where(distances <= distance_threshold)[0]
        inlier_count = len(inliers_local)
        
        # 更新最佳模型
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d
            best_inliers = local_region[inliers_local]
            
            logger.debug(f"迭代 {i}: 内点数量 = {inlier_count}, 内点比例 = {inlier_count/len(local_region):.2%}")
            
            # 更新所需迭代次数
            if inlier_count > 0:
                w = inlier_count / len(local_region)
                if w > 0:
                    iterations = min(max_iterations, int(np.log(1 - confidence) / np.log(1 - w**3)))
    
    logger.info(f"RANSAC平面提取完成: 找到 {best_inlier_count} 个内点，占局部区域的 {best_inlier_count/len(local_region):.2%}")
    
    return best_inliers, best_normal, best_d


# ============================================================================# 区域增长平面提取
# ============================================================================_
def region_growing_plane_extraction(
    points: np.ndarray,
    reference_point: np.ndarray,
    target_normal: np.ndarray,
    local_region: np.ndarray,
    normal_threshold: float = 10.0,
    distance_threshold: float = 0.01,
    num_neighbors: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用区域增长方法提取平面
    
    Args:
        points: N×3点云数组
        reference_point: 参考点 p0
        target_normal: 目标法向量
        local_region: 局部搜索区域内的点索引
        normal_threshold: 法向量夹角阈值（度）
        distance_threshold: 点到平面的距离阈值
        num_neighbors: K近邻数量
    
    Returns:
        (region_indices, region_normal): 区域内的点索引和区域法向量
    """
    logger = get_logger()
    logger.info("开始区域增长平面提取")
    
    # 获取局部区域内的点
    local_points = points[local_region]
    
    # 构建KD树用于近邻搜索
    kdtree = KDTree(local_points)
    
    # 选择种子点：参考点的最近邻点
    distances = np.linalg.norm(local_points - reference_point, axis=1)
    seed_idx_local = np.argmin(distances)
    seed_idx_global = local_region[seed_idx_local]
    
    logger.debug(f"选择种子点: 局部索引={seed_idx_local}, 全局索引={seed_idx_global}")
    
    # 计算种子点的法向量
    # 获取种子点的K近邻
    _, neighbors_local = kdtree.query(local_points[seed_idx_local], k=num_neighbors+1)
    neighbors_local = neighbors_local[1:]  # 排除自身
    
    # 使用PCA计算法向量
    neighbor_points = local_points[neighbors_local]
    centroid = np.mean(neighbor_points, axis=0)
    covariance = np.cov(neighbor_points - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # 最小特征值对应的特征向量即为法向量
    normal = eigenvectors[:, 0]
    
    # 确保法向量与目标法向量方向一致
    if np.dot(normal, target_normal) < 0:
        normal = -normal
    
    # 检查法向量是否与目标法向量一致
    if not check_normal_consistency(normal, target_normal, angle_threshold=30.0):
        logger.warning("种子点法向量与目标法向量不一致，无法进行区域增长")
        return np.array([]), np.zeros(3)
    
    logger.debug(f"种子点法向量: {normal}")
    
    # 区域增长
    visited = np.zeros(len(local_points), dtype=bool)
    region = []
    queue = [seed_idx_local]
    
    while queue:
        current_idx_local = queue.pop(0)
        
        if visited[current_idx_local]:
            continue
        
        visited[current_idx_local] = True
        region.append(current_idx_local)
        
        # 获取当前点的K近邻
        _, neighbors_local = kdtree.query(local_points[current_idx_local], k=num_neighbors+1)
        neighbors_local = neighbors_local[1:]  # 排除自身
        
        for neighbor_idx_local in neighbors_local:
            if visited[neighbor_idx_local]:
                continue
            
            # 计算邻居点的法向量
            _, neighbor_neighbors = kdtree.query(local_points[neighbor_idx_local], k=num_neighbors+1)
            neighbor_neighbors = neighbor_neighbors[1:]  # 排除自身
            
            if len(neighbor_neighbors) < 3:
                continue
            
            # 使用PCA计算邻居点的法向量
            neighbor_points = local_points[neighbor_neighbors]
            centroid = np.mean(neighbor_points, axis=0)
            covariance = np.cov(neighbor_points - centroid, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            
            neighbor_normal = eigenvectors[:, 0]
            
            # 确保法向量与目标法向量方向一致
            if np.dot(neighbor_normal, target_normal) < 0:
                neighbor_normal = -neighbor_normal
            
            # 检查法向量一致性
            if check_normal_consistency(neighbor_normal, normal, angle_threshold=normal_threshold):
                queue.append(neighbor_idx_local)
    
    # 转换为全局索引
    region_indices = local_region[region]
    
    logger.info(f"区域增长平面提取完成: 找到 {len(region)} 个点，占局部区域的 {len(region)/len(local_region):.2%}")
    
    return region_indices, normal


# ============================================================================# 主函数：extract_plane
# ============================================================================_
def extract_plane(
    points: np.ndarray,
    ratio_point: np.ndarray,
    target_normal: np.ndarray,
    method: str = "ransac",
    **kwargs
) -> np.ndarray:
    """
    面提取函数
    
    Args:
        points: N×3的三维坐标数组
        ratio_point: 比例参考点 [rx, ry, rz]，表示在全局包围盒中的相对位置
        target_normal: 大致法向量，指示目标平面的朝向
        method: 平面拟合方法，可选 "ransac" 或 "region_growing"
        **kwargs: 传递给具体拟合方法的参数
    
    Returns:
        属于目标平面的原始点索引
    """
    logger = get_logger()
    logger.info("开始面提取")
    
    # 参数验证
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数据必须是N×3的三维坐标数组")
    
    if len(ratio_point) != 3:
        raise ValueError("比例参考点必须是长度为3的数组")
    
    if len(target_normal) != 3:
        raise ValueError("大致法向量必须是长度为3的数组")
    
    # 归一化目标法向量
    target_normal = target_normal / np.linalg.norm(target_normal)
    
    logger.info(f"点云数量: {len(points)}")
    logger.info(f"比例参考点: {ratio_point}")
    logger.info(f"目标法向量: {target_normal}")
    logger.info(f"拟合方法: {method}")
    
    # 获取参考点
    reference_point = get_reference_point(points, ratio_point)
    logger.info(f"实际参考点位置: {reference_point}")
    
    # 获取局部搜索区域
    local_region = get_local_region(points, reference_point)
    logger.info(f"局部搜索区域点数量: {len(local_region)}")
    
    # 根据选择的方法进行平面提取
    if method == "ransac":
        inliers, _, _ = ransac_plane_extraction(
            points, reference_point, target_normal, local_region, **kwargs
        )
        logger.info(f"RANSAC提取到 {len(inliers)} 个平面点")
        return inliers
    elif method == "region_growing":
        region_indices, _ = region_growing_plane_extraction(
            points, reference_point, target_normal, local_region, **kwargs
        )
        logger.info(f"区域增长提取到 {len(region_indices)} 个平面点")
        return region_indices
    else:
        raise ValueError(f"未知的拟合方法: {method}，可选方法: ransac, region_growing")


# ============================================================================# 示例代码
# ============================================================================_
if __name__ == "__main__":
    # 初始化日志
    logger = setup_logger(log_dir="./logs", console_output=True)
    
    print("点云平面提取模块示例")
    
    # 创建测试数据
    np.random.seed(42)
    
    # 生成平面点云
    n_plane = 10000
    x = np.random.rand(n_plane) * 10
    y = np.random.rand(n_plane) * 10
    z = 0.5 * x + 0.3 * y + 2.0  # 平面方程: 0.5x + 0.3y - z + 2.0 = 0
    plane_points = np.column_stack([x, y, z])
    
    # 添加噪声
    plane_points += np.random.randn(n_plane, 3) * 0.1
    
    # 添加离群点
    n_outliers = 2000
    outliers = np.random.rand(n_outliers, 3) * 20 - np.array([5, 5, 5])
    
    # 合并点云
    all_points = np.vstack([plane_points, outliers])
    
    print(f"测试点云: {len(all_points)} 个点")
    
    # 提取平面
    ratio_point = np.array([0.5, 0.5, 0.5])  # 中心位置
    target_normal = np.array([-0.5, -0.3, 1.0])  # 平面法向量
    
    print("\n使用RANSAC方法提取平面...")
    plane_indices = extract_plane(
        all_points,
        ratio_point=ratio_point,
        target_normal=target_normal,
        method="ransac",
        distance_threshold=0.2,
        max_iterations=1000
    )
    
    print(f"RANSAC提取到 {len(plane_indices)} 个平面点")
    
    print("\n使用区域增长方法提取平面...")
    plane_indices_rg = extract_plane(
        all_points,
        ratio_point=ratio_point,
        target_normal=target_normal,
        method="region_growing",
        normal_threshold=15.0,
        distance_threshold=0.2,
        num_neighbors=20
    )
    
    print(f"区域增长提取到 {len(plane_indices_rg)} 个平面点")
    
    print("\n示例完成")
    logger.info("示例程序执行完成")
