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
import logging
import numpy as np
import cv2
from datetime import datetime
from typing import Tuple, Optional


# ============================================================================
# 日志配置（与pointcloud_loading.py共用）
# ============================================================================

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
    log_file = os.path.join(log_dir, f"depth_render_{timestamp}.log")
    
    logger = logging.getLogger("DepthMapRenderer")
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
        resolution: 分辨率（单位长度对应像素数）
        output_file: 输出路径（可选）
    
    Returns:
        (depth_map, depth_image_colored): 原始深度图 和 彩色可视化图
    """
    logger = get_logger()
    
    # 数据验证
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数据必须是n×3的数组")
    
    logger.info(f"生成深度图: {len(points)} 点, 分辨率={resolution}")
    
    x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    width = int((x_max - x_min) * resolution + 1)
    height = int((y_max - y_min) * resolution + 1)
    
    logger.info(f"尺寸: {width}×{height} px")
    logger.info(f"X:[{x_min:.4f},{x_max:.4f}] Y:[{y_min:.4f},{y_max:.4f}] Z:[{z_coords.min():.4f},{z_coords.max():.4f}]")
    
    # 点云投影
    depth_map = _project_points_to_depth(points, x_min, y_min, resolution, width, height)
    
    # 着色
    depth_image_colored = _colorize_depth_map(depth_map)
    
    # 保存
    if output_file:
        _save_depth_outputs(depth_map, depth_image_colored, output_file)
        print(f"深度图已保存: {output_file}")
    
    logger.info("深度图生成完成")
    return depth_map, depth_image_colored


def _project_points_to_depth(
    points: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    width: int,
    height: int
) -> np.ndarray:
    """点云投影到深度图（近距离优先）"""
    logger = get_logger()
    
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    
    px = ((points[:, 0] - x_min) * resolution).astype(np.int32)
    py = ((points[:, 1] - y_min) * resolution).astype(np.int32)
    z = points[:, 2]
    
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, z = px[valid], py[valid], z[valid]
    
    logger.debug(f"有效投影点: {len(px)}/{len(points)}")
    
    # 按z排序（大→小），小z后写入覆盖大z
    sort_idx = np.argsort(-z)
    px, py, z = px[sort_idx], py[sort_idx], z[sort_idx]
    
    depth_map[py, px] = z
    depth_map[depth_map == np.inf] = np.nan
    
    valid_count = np.sum(~np.isnan(depth_map))
    logger.info(f"投影完成: 有效像素={valid_count}, 覆盖率={100*valid_count/(width*height):.1f}%")
    
    return depth_map


# ============================================================================
# 改进Otsu阈值算法
# ============================================================================

def otsu_threshold_improved(values: np.ndarray, num_bins: int = 256) -> float:
    """
    改进Otsu阈值算法
    
    改进：1.直方图修剪去噪 2.阈值细化（方差相近选中心）
    """
    logger = get_logger()
    
    if len(values) == 0:
        return 0.0
    
    hist, bin_edges = np.histogram(values, bins=num_bins)
    
    # 直方图修剪
    non_zero = np.where(hist > 0)[0]
    if len(non_zero) == 0:
        return np.median(values)
    
    start_idx, end_idx = non_zero[0], non_zero[-1]
    trimmed_hist = hist[start_idx:end_idx + 1]
    
    logger.debug(f"Otsu修剪: bins [{start_idx},{end_idx}]")
    
    if len(trimmed_hist) <= 1:
        return np.median(values)
    
    total = trimmed_hist.sum()
    if total == 0:
        return np.median(values)
    
    level_indices = np.arange(len(trimmed_hist))
    global_mean = np.dot(level_indices, trimmed_hist) / total
    
    max_variance = 0.0
    best_idx = 0
    cumulative_sum = 0.0
    cumulative_mean = 0.0
    
    for i in range(len(trimmed_hist)):
        cumulative_sum += trimmed_hist[i]
        cumulative_mean += i * trimmed_hist[i]
        
        w0, w1 = cumulative_sum / total, 1.0 - cumulative_sum / total
        
        if w0 <= 0 or w1 <= 0:
            continue
        
        mu0 = cumulative_mean / cumulative_sum
        mu1 = (global_mean * total - cumulative_mean) / (total - cumulative_sum)
        
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        center = len(trimmed_hist) / 2
        if variance > max_variance or \
           (np.isclose(variance, max_variance, rtol=1e-6) and abs(i - center) < abs(best_idx - center)):
            max_variance = variance
            best_idx = i
    
    threshold = bin_edges[start_idx + best_idx]
    logger.info(f"Otsu阈值: {threshold:.4f}, 方差: {max_variance:.4f}")
    
    return threshold


# ============================================================================
# 深度图着色
# ============================================================================

def _colorize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """深度图着色：前景JET+CLAHE，背景黑灰渐变"""
    logger = get_logger()
    
    valid_mask = ~np.isnan(depth_map)
    
    if not np.any(valid_mask):
        logger.warning("深度图无有效数据")
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    
    valid_depths = depth_map[valid_mask]
    
    depth_min = np.percentile(valid_depths, 1)
    depth_max = np.percentile(valid_depths, 99)
    
    logger.debug(f"深度范围(1%-99%): [{depth_min:.4f}, {depth_max:.4f}]")
    
    # Otsu分割
    otsu_thresh = otsu_threshold_improved(valid_depths[valid_depths >= depth_min], 256)
    p85 = np.percentile(valid_depths, 85)
    fg_upper = min(otsu_thresh, p85)
    
    logger.info(f"分割阈值: Otsu={otsu_thresh:.4f}, P85={p85:.4f}, 使用={fg_upper:.4f}")
    
    # 掩码
    fg_mask = (depth_map <= fg_upper) & valid_mask
    bg_mask = valid_mask & ~fg_mask
    
    logger.info(f"前景占比: {100*np.sum(fg_mask)/np.sum(valid_mask):.1f}%")
    
    colored = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    
    # 前景: JET + γ校正 + CLAHE
    if np.any(fg_mask):
        fg_norm = np.clip(depth_map[fg_mask], depth_min, fg_upper)
        fg_norm = (fg_norm - depth_min) / (fg_upper - depth_min + 1e-8)
        fg_gamma = np.power(fg_norm, 0.7)
        
        fg_8bit = np.zeros_like(depth_map, dtype=np.uint8)
        fg_8bit[fg_mask] = (255 * fg_gamma).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        fg_enhanced = clahe.apply(fg_8bit)
        fg_colored = cv2.applyColorMap(fg_enhanced, cv2.COLORMAP_JET)
        colored[fg_mask] = fg_colored[fg_mask]
    
    # 背景: 黑灰渐变
    if np.any(bg_mask):
        bg_norm = np.clip(depth_map[bg_mask], fg_upper, depth_max)
        bg_norm = (bg_norm - fg_upper) / (depth_max - fg_upper + 1e-8)
        bg_gray = (30 + 50 * bg_norm).astype(np.uint8)
        
        colored[bg_mask, 0] = bg_gray
        colored[bg_mask, 1] = bg_gray
        colored[bg_mask, 2] = bg_gray
    
    return colored


# ============================================================================
# 输出保存
# ============================================================================

def _save_depth_outputs(depth_map: np.ndarray, colored: np.ndarray, output_file: str):
    """保存深度图"""
    logger = get_logger()
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    cv2.imwrite(output_file, colored)
    logger.info(f"彩色图保存: {output_file}")
    
    npy_file = os.path.splitext(output_file)[0] + '_depth.npy'
    np.save(npy_file, depth_map)
    logger.info(f"深度数据保存: {npy_file}")


# ============================================================================
# 辅助函数
# ============================================================================

def depth_map_to_pointcloud(
    depth_map: np.ndarray,
    resolution: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0
) -> np.ndarray:
    """深度图逆变换回点云"""
    valid_mask = ~np.isnan(depth_map)
    py, px = np.where(valid_mask)
    z = depth_map[valid_mask]
    
    x = px / resolution + x_offset
    y = py / resolution + y_offset
    
    return np.column_stack([x, y, z])


def compute_depth_statistics(depth_map: np.ndarray) -> dict:
    """计算深度图统计信息"""
    logger = get_logger()
    
    valid_mask = ~np.isnan(depth_map)
    valid_depths = depth_map[valid_mask]
    
    if len(valid_depths) == 0:
        return {'valid_pixels': 0}
    
    stats = {
        'valid_pixels': int(np.sum(valid_mask)),
        'total_pixels': depth_map.size,
        'coverage': float(np.sum(valid_mask) / depth_map.size),
        'min': float(valid_depths.min()),
        'max': float(valid_depths.max()),
        'mean': float(valid_depths.mean()),
        'std': float(valid_depths.std()),
        'median': float(np.median(valid_depths)),
    }
    
    logger.debug(f"统计: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}")
    return stats


def batch_generate_depth_maps(
    input_files: list,
    output_dir: str,
    resolution: float = 1.0,
    load_func=None
) -> list:
    """批量生成深度图"""
    logger = get_logger()
    
    os.makedirs(output_dir, exist_ok=True)
    if load_func is None:
        load_func = np.loadtxt
    
    output_files = []
    total = len(input_files)
    
    print(f"批量处理 {total} 个文件...")
    logger.info(f"批量处理: {total} 文件, 输出={output_dir}")
    
    for i, f in enumerate(input_files):
        logger.info(f"[{i+1}/{total}] {f}")
        try:
            points = load_func(f)
            out = os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + '_depth.png')
            generate_depth_map(points, resolution, out)
            output_files.append(out)
        except Exception as e:
            logger.error(f"失败 {f}: {e}")
    
    print(f"完成: {len(output_files)}/{total}")
    return output_files


# ============================================================================
# 示例
# ============================================================================

if __name__ == "__main__":
    logger = setup_logger(log_dir="./logs", console_output=False)
    
    # 测试数据
    np.random.seed(42)
    n = 10000
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = 10 * np.exp(-((x-50)**2 + (y-50)**2) / 500) + np.random.normal(0, 0.5, n)
    
    depth_map, colored = generate_depth_map(np.column_stack([x, y, z]), 2.0, "./test_depth.png")
    compute_depth_statistics(depth_map)
    
    print("完成，详情见 ./logs/")