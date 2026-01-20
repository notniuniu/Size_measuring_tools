#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于点云投影的深度图渲染模块 (Depth Map Rendering from Point Cloud Projection)

核心算法：
1. 点云→深度图投影（近距离优先策略处理遮挡）
2. 改进Otsu阈值算法（直方图修剪+阈值细化）自动分割前景背景
3. 差异化着色：前景JET映射+CLAHE增强，背景黑灰渐变

输出：彩色深度图(.png) + 原始深度数据(.npy)
"""

import os
import numpy as np
import cv2
from typing import Tuple, Optional


# ============================================================================
# 核心函数：深度图生成
# ============================================================================

def generate_depth_map(
    points: np.ndarray,
    resolution: float = 1.0,
    output_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从点云数据生成深度图
    
    原理：将点云投影到xy平面，z值作为深度
    遮挡处理：同一像素位置保留最小z值（近距离优先）
    
    Args:
        points: n×3 numpy数组 [x, y, z]
        resolution: 分辨率（单位长度对应像素数），值越大图像越精细
        output_file: 输出路径（可选）
    
    Returns:
        (depth_map, depth_image_colored): 原始深度图(含NaN) 和 彩色可视化图
    """
    # -------------------------
    # 1. 数据验证与预处理
    # -------------------------
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数据必须是n×3的数组")
    
    x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
    
    # 计算xy平面边界
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 计算深度图尺寸
    width = int((x_max - x_min) * resolution + 1)
    height = int((y_max - y_min) * resolution + 1)
    
    print(f"[深度图] 尺寸: {width}×{height} px, 分辨率: {resolution}")
    print(f"[深度图] X范围: [{x_min:.2f}, {x_max:.2f}], Y范围: [{y_min:.2f}, {y_max:.2f}]")
    
    # -------------------------
    # 2. 点云投影（近距离优先）
    # -------------------------
    depth_map = _project_points_to_depth(
        points, x_min, y_min, resolution, width, height
    )
    
    # -------------------------
    # 3. 深度值分析与着色
    # -------------------------
    depth_image_colored = _colorize_depth_map(depth_map)
    
    # -------------------------
    # 4. 保存结果
    # -------------------------
    if output_file:
        _save_depth_outputs(depth_map, depth_image_colored, output_file)
    
    return depth_map, depth_image_colored


def _project_points_to_depth(
    points: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    width: int,
    height: int
) -> np.ndarray:
    """
    点云投影到深度图（向量化实现）
    
    映射公式：
        px = int((x - x_min) * resolution)
        py = int((y - y_min) * resolution)
    
    遮挡处理：同一像素保留最小z值（最近点可见）
    """
    # 初始化为无穷大
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    
    # 计算像素坐标
    px = ((points[:, 0] - x_min) * resolution).astype(np.int32)
    py = ((points[:, 1] - y_min) * resolution).astype(np.int32)
    z = points[:, 2]
    
    # 边界检查
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, z = px[valid], py[valid], z[valid]
    
    # 按z值排序（从大到小），确保小z值最后写入（覆盖大值）
    sort_idx = np.argsort(-z)
    px, py, z = px[sort_idx], py[sort_idx], z[sort_idx]
    
    # 写入深度图（后写入的小z会覆盖先写入的大z）
    depth_map[py, px] = z
    
    # 无效位置标记为NaN
    depth_map[depth_map == np.inf] = np.nan
    
    valid_count = np.sum(~np.isnan(depth_map))
    print(f"[深度图] 有效像素: {valid_count} ({100*valid_count/(width*height):.1f}%)")
    
    return depth_map


# ============================================================================
# 改进Otsu阈值算法
# ============================================================================

def otsu_threshold_improved(values: np.ndarray, num_bins: int = 256) -> float:
    """
    改进的Otsu阈值算法
    
    改进点：
    1. 直方图修剪：去除两端零值区间，降低噪声干扰
    2. 阈值细化：方差相近时优先选择靠近中心的阈值
    
    原理：
    - 将数据按阈值t分为前景C0和背景C1
    - 最大化类间方差 σ²_b = ω0·ω1·(μ0-μ1)²
    - 最优阈值 t* = argmax(σ²_b)
    
    Args:
        values: 有效深度值数组
        num_bins: 直方图bin数量
    
    Returns:
        最优阈值对应的深度值
    """
    if len(values) == 0:
        return 0.0
    
    # 构建直方图
    hist, bin_edges = np.histogram(values, bins=num_bins)
    
    # ---- 改进1：直方图修剪 ----
    non_zero = np.where(hist > 0)[0]
    if len(non_zero) == 0:
        return np.median(values)
    
    start_idx, end_idx = non_zero[0], non_zero[-1]
    trimmed_hist = hist[start_idx:end_idx + 1]
    
    if len(trimmed_hist) <= 1:
        return np.median(values)
    
    # ---- Otsu核心计算 ----
    total = trimmed_hist.sum()
    if total == 0:
        return np.median(values)
    
    # 计算全局均值
    level_indices = np.arange(len(trimmed_hist))
    global_mean = np.dot(level_indices, trimmed_hist) / total
    
    # 遍历所有阈值，寻找最大类间方差
    max_variance = 0.0
    best_threshold_idx = 0
    
    cumulative_sum = 0.0
    cumulative_mean = 0.0
    
    for i in range(len(trimmed_hist)):
        cumulative_sum += trimmed_hist[i]
        cumulative_mean += i * trimmed_hist[i]
        
        w0 = cumulative_sum / total  # 前景权重
        w1 = 1.0 - w0                 # 背景权重
        
        if w0 <= 0 or w1 <= 0:
            continue
        
        mu0 = cumulative_mean / cumulative_sum  # 前景均值
        mu1 = (global_mean * total - cumulative_mean) / (total - cumulative_sum)  # 背景均值
        
        # 类间方差
        between_variance = w0 * w1 * (mu0 - mu1) ** 2
        
        # ---- 改进2：阈值细化 ----
        # 方差相近时优先选择更靠近中心的阈值
        center = len(trimmed_hist) / 2
        if between_variance > max_variance or \
           (np.isclose(between_variance, max_variance, rtol=1e-6) and
            abs(i - center) < abs(best_threshold_idx - center)):
            max_variance = between_variance
            best_threshold_idx = i
    
    # 映射回原始深度值
    original_idx = start_idx + best_threshold_idx
    threshold_value = bin_edges[original_idx]
    
    return threshold_value


# ============================================================================
# 深度图着色
# ============================================================================

def _colorize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    深度图差异化着色
    
    着色策略：
    - 前景（近处）：JET色彩映射 + 伽马校正 + CLAHE增强
    - 背景（远处）：黑灰色渐变
    
    分割方法：改进Otsu阈值自动确定前景/背景边界
    """
    valid_mask = ~np.isnan(depth_map)
    
    if not np.any(valid_mask):
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    
    valid_depths = depth_map[valid_mask]
    
    # -------------------------
    # 深度范围与阈值计算
    # -------------------------
    # 去除极端值（1%~99%分位数）
    depth_min = np.percentile(valid_depths, 1)
    depth_max = np.percentile(valid_depths, 99)
    
    # Otsu阈值确定前景上限
    otsu_threshold = otsu_threshold_improved(
        valid_depths[valid_depths >= depth_min],
        num_bins=256
    )
    
    # 结合85%分位数作为保护机制
    percentile_85 = np.percentile(valid_depths, 85)
    foreground_upper = min(otsu_threshold, percentile_85)
    
    print(f"[着色] Otsu阈值: {otsu_threshold:.4f}, 85%分位: {percentile_85:.4f}")
    print(f"[着色] 前景范围: [{depth_min:.4f}, {foreground_upper:.4f}]")
    
    # -------------------------
    # 创建前景/背景掩码
    # -------------------------
    foreground_mask = (depth_map <= foreground_upper) & valid_mask
    background_mask = valid_mask & ~foreground_mask
    
    fg_ratio = 100 * np.sum(foreground_mask) / np.sum(valid_mask)
    print(f"[着色] 前景占比: {fg_ratio:.1f}%")
    
    # 初始化输出图像
    colored_image = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    
    # -------------------------
    # 前景着色：JET + 伽马校正 + CLAHE
    # -------------------------
    if np.any(foreground_mask):
        fg_depths = depth_map[foreground_mask]
        
        # 归一化到[0,1]
        fg_normalized = np.clip(fg_depths, depth_min, foreground_upper)
        fg_normalized = (fg_normalized - depth_min) / (foreground_upper - depth_min + 1e-8)
        
        # 伽马校正（γ<1增强暗部细节）
        gamma = 0.7
        fg_gamma = np.power(fg_normalized, gamma)
        
        # 转换为8位图像用于CLAHE
        fg_8bit = np.zeros_like(depth_map, dtype=np.uint8)
        fg_8bit[foreground_mask] = (255 * fg_gamma).astype(np.uint8)
        
        # CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        fg_enhanced = clahe.apply(fg_8bit)
        
        # JET色彩映射
        fg_colored = cv2.applyColorMap(fg_enhanced, cv2.COLORMAP_JET)
        colored_image[foreground_mask] = fg_colored[foreground_mask]
    
    # -------------------------
    # 背景着色：黑灰渐变
    # -------------------------
    if np.any(background_mask):
        bg_depths = depth_map[background_mask]
        
        # 归一化到[30, 80]灰度范围
        bg_normalized = np.clip(bg_depths, foreground_upper, depth_max)
        bg_normalized = (bg_normalized - foreground_upper) / (depth_max - foreground_upper + 1e-8)
        bg_gray = (30 + 50 * bg_normalized).astype(np.uint8)
        
        # 灰度转BGR
        colored_image[background_mask, 0] = bg_gray
        colored_image[background_mask, 1] = bg_gray
        colored_image[background_mask, 2] = bg_gray
    
    return colored_image


# ============================================================================
# 输出保存
# ============================================================================

def _save_depth_outputs(
    depth_map: np.ndarray,
    colored_image: np.ndarray,
    output_file: str
):
    """保存彩色深度图和原始深度数据"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # 保存彩色图像
    cv2.imwrite(output_file, colored_image)
    print(f"[保存] 彩色深度图: {output_file}")
    
    # 保存原始深度数据(.npy)
    npy_file = os.path.splitext(output_file)[0] + '_depth.npy'
    np.save(npy_file, depth_map)
    print(f"[保存] 原始深度数据: {npy_file}")


# ============================================================================
# 辅助函数
# ============================================================================

def depth_map_to_pointcloud(
    depth_map: np.ndarray,
    resolution: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0
) -> np.ndarray:
    """
    深度图逆变换回点云（用于验证或重建）
    
    Args:
        depth_map: 深度图数组
        resolution: 原始分辨率
        x_offset, y_offset: 原始x_min, y_min
    
    Returns:
        n×3点云数组
    """
    valid_mask = ~np.isnan(depth_map)
    py, px = np.where(valid_mask)
    z = depth_map[valid_mask]
    
    x = px / resolution + x_offset
    y = py / resolution + y_offset
    
    return np.column_stack([x, y, z])


def compute_depth_statistics(depth_map: np.ndarray) -> dict:
    """计算深度图统计信息"""
    valid_mask = ~np.isnan(depth_map)
    valid_depths = depth_map[valid_mask]
    
    if len(valid_depths) == 0:
        return {'valid_pixels': 0}
    
    return {
        'valid_pixels': int(np.sum(valid_mask)),
        'total_pixels': depth_map.size,
        'coverage': float(np.sum(valid_mask) / depth_map.size),
        'min': float(valid_depths.min()),
        'max': float(valid_depths.max()),
        'mean': float(valid_depths.mean()),
        'std': float(valid_depths.std()),
        'median': float(np.median(valid_depths)),
        'p5': float(np.percentile(valid_depths, 5)),
        'p95': float(np.percentile(valid_depths, 95)),
    }


# ============================================================================
# 批量处理
# ============================================================================

def batch_generate_depth_maps(
    input_files: list,
    output_dir: str,
    resolution: float = 1.0,
    load_func=None
) -> list:
    """
    批量生成深度图
    
    Args:
        input_files: 输入点云文件列表
        output_dir: 输出目录
        resolution: 分辨率
        load_func: 点云加载函数，默认np.loadtxt
    
    Returns:
        输出文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if load_func is None:
        load_func = lambda f: np.loadtxt(f)
    
    output_files = []
    
    for i, input_file in enumerate(input_files):
        print(f"\n[{i+1}/{len(input_files)}] 处理: {input_file}")
        
        try:
            points = load_func(input_file)
            
            basename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, f"{basename}_depth.png")
            
            generate_depth_map(points, resolution, output_file)
            output_files.append(output_file)
            
        except Exception as e:
            print(f"[错误] {input_file}: {e}")
    
    print(f"\n[完成] 共处理 {len(output_files)}/{len(input_files)} 个文件")
    return output_files


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建测试点云（模拟地形）
    np.random.seed(42)
    n_points = 10000
    
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 100, n_points)
    # z为地形高度，模拟一个山丘
    z = 10 * np.exp(-((x-50)**2 + (y-50)**2) / 500) + np.random.normal(0, 0.5, n_points)
    
    test_points = np.column_stack([x, y, z])
    
    # 生成深度图
    depth_map, colored_image = generate_depth_map(
        test_points,
        resolution=2.0,
        output_file="./test_depth.png"
    )
    
    # 打印统计信息
    stats = compute_depth_statistics(depth_map)
    print("\n深度图统计:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")