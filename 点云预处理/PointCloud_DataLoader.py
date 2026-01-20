#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云加载工具模块 (Pointcloud Loading Utilities)
支持格式: PLY, PCD, TXT(ASCII/Binary), XYZ, BIN
输出格式: Open3D PointCloud
功能: 加载、降采样、可视化、日志记录
"""

import os
import struct
import logging
import numpy as np
import open3d as o3d
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Union

# ============================================================================
# 日志配置
# ============================================================================

def setup_logger(
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        log_dir: 日志文件目录
        log_level: 日志级别
        console_output: 是否输出到控制台
    
    Returns:
        配置好的Logger对象
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pointcloud_{timestamp}.log")
    
    logger = logging.getLogger("PointCloudLoader")
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    return logger

# 默认日志器
_logger: Optional[logging.Logger] = None

def get_logger() -> logging.Logger:
    """获取全局日志器，未初始化则自动创建"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger

def set_logger(logger: logging.Logger):
    """设置自定义日志器"""
    global _logger
    _logger = logger


# ============================================================================
# 核心加载函数
# ============================================================================

def load_pointcloud(
    file_path: str,
    file_format: Optional[str] = None,
    binary_record_format: str = '3fi',
    skip_header: int = 0,
    delimiter: str = None,
    xyz_columns: Tuple[int, int, int] = (0, 1, 2)
) -> o3d.geometry.PointCloud:
    """
    统一点云加载接口，自动识别格式
    
    Args:
        file_path: 文件路径
        file_format: 强制指定格式 ('ply','pcd','txt','txt_bin','xyz','bin')，None则自动检测
        binary_record_format: 二进制TXT的struct格式，默认'3fi'(3个float+1个int)
        skip_header: TXT文件跳过的头部行数
        delimiter: TXT分隔符，None则自动检测
        xyz_columns: TXT中xyz对应的列索引
    
    Returns:
        Open3D PointCloud对象
    """
    logger = get_logger()
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    file_size = os.path.getsize(file_path) / 1024 / 1024
    logger.info(f"加载文件: {file_path} ({file_size:.2f} MB)")
    
    # 自动检测格式
    if file_format is None:
        file_format = _detect_format(file_path)
    file_format = file_format.lower()
    logger.info(f"文件格式: {file_format}")
    
    # 根据格式调用对应加载函数
    loaders = {
        'ply': lambda: _load_ply(file_path),
        'pcd': lambda: _load_pcd(file_path),
        'xyz': lambda: _load_txt_ascii(file_path, skip_header, delimiter, xyz_columns),
        'txt': lambda: _load_txt_ascii(file_path, skip_header, delimiter, xyz_columns),
        'txt_bin': lambda: _load_txt_binary(file_path, binary_record_format),
        'bin': lambda: _load_txt_binary(file_path, binary_record_format),
    }
    
    if file_format not in loaders:
        logger.error(f"不支持的格式: {file_format}")
        raise ValueError(f"不支持的格式: {file_format}，支持: {list(loaders.keys())}")
    
    pcd = loaders[file_format]()
    logger.info(f"加载完成: {len(pcd.points)} 个点")
    log_pointcloud_stats(pcd)
    
    return pcd


def _detect_format(file_path: str) -> str:
    """根据扩展名和内容检测文件格式"""
    ext = Path(file_path).suffix.lower()
    
    if ext in ['.ply']:
        return 'ply'
    elif ext in ['.pcd']:
        return 'pcd'
    elif ext in ['.xyz']:
        return 'xyz'
    elif ext in ['.bin']:
        return 'bin'
    elif ext in ['.txt', '']:
        # 检测txt是否为二进制
        return 'txt_bin' if _is_binary_file(file_path) else 'txt'
    else:
        return 'txt'  # 默认当作ASCII txt处理


def _is_binary_file(file_path: str, check_bytes: int = 8192) -> bool:
    """检测文件是否为二进制格式"""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(check_bytes)
        # 二进制文件通常包含空字节
        return b'\x00' in chunk
    except:
        return False


# ============================================================================
# 各格式加载实现
# ============================================================================

def _load_ply(file_path: str) -> o3d.geometry.PointCloud:
    """加载PLY格式"""
    return o3d.io.read_point_cloud(file_path)


def _load_pcd(file_path: str) -> o3d.geometry.PointCloud:
    """加载PCD格式"""
    return o3d.io.read_point_cloud(file_path)


def _load_txt_ascii(
    file_path: str,
    skip_header: int = 0,
    delimiter: str = None,
    xyz_columns: Tuple[int, int, int] = (0, 1, 2)
) -> o3d.geometry.PointCloud:
    """
    加载ASCII格式的TXT/XYZ点云
    
    支持格式示例:
        x y z
        x,y,z
        x y z r g b
    """
    logger = get_logger()
    
    try:
        # 自动检测分隔符
        if delimiter is None:
            with open(file_path, 'r') as f:
                for _ in range(skip_header):
                    f.readline()
                first_line = f.readline().strip()
            delimiter = ',' if ',' in first_line else None  # None表示空白符
        
        # 加载数据
        data = np.loadtxt(
            file_path,
            delimiter=delimiter,
            skiprows=skip_header,
            dtype=np.float64
        )
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # 提取XYZ
        x_col, y_col, z_col = xyz_columns
        points = data[:, [x_col, y_col, z_col]]
        
        # 过滤无效点
        valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
        points = points[valid_mask]
        
        if np.sum(~valid_mask) > 0:
            logger.warning(f"过滤无效点: {np.sum(~valid_mask)} 个")
        
        # 转换为Open3D格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
        
    except Exception as e:
        logger.error(f"加载TXT失败: {e}")
        raise


def _load_txt_binary(
    file_path: str,
    record_format: str = '3fi'
) -> o3d.geometry.PointCloud:
    """
    加载二进制格式的点云文件
    
    Args:
        file_path: 文件路径
        record_format: struct格式字符串
            '3f'  - 仅xyz (12字节)
            '3fi' - xyz + int (16字节)
            '6f'  - xyz + rgb (24字节)
    """
    logger = get_logger()
    
    try:
        record_size = struct.calcsize(record_format)
        file_size = os.path.getsize(file_path)
        total_records = file_size // record_size
        
        if file_size % record_size != 0:
            logger.warning(f"文件大小非整数倍，尾部 {file_size % record_size} 字节被忽略")
        
        if total_records == 0:
            logger.error("无有效记录")
            return o3d.geometry.PointCloud()
        
        # 构建numpy dtype
        dtype_map = {
            '3f': [('x', 'f4'), ('y', 'f4'), ('z', 'f4')],
            '3fi': [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('_', 'i4')],
            '6f': [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'f4'), ('g', 'f4'), ('b', 'f4')],
            '3fI': [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('_', 'u4')],
        }
        
        if record_format in dtype_map:
            dtype = np.dtype(dtype_map[record_format]).newbyteorder('<')
        else:
            # 通用解析：假设前3个float为xyz
            logger.warning(f"非标准格式 {record_format}，仅提取前3个float作为xyz")
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            points = []
            for i in range(total_records):
                offset = i * record_size
                record = struct.unpack(record_format, raw_data[offset:offset+record_size])
                points.append(record[:3])
            points = np.array(points, dtype=np.float64)
            
            valid_mask = ~np.isnan(points[:, 2])
            points = points[valid_mask]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            return pcd
        
        # 高效批量读取
        data = np.fromfile(file_path, dtype=dtype, count=total_records)
        points = np.column_stack([data['x'], data['y'], data['z']])
        
        # 过滤无效点
        valid_mask = ~np.isnan(points[:, 2])
        points = points[valid_mask].astype(np.float64)
        
        if np.sum(~valid_mask) > 0:
            logger.warning(f"过滤NaN点: {np.sum(~valid_mask)} 个")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
        
    except Exception as e:
        logger.error(f"加载二进制文件失败: {e}")
        raise


# ============================================================================
# 降采样
# ============================================================================

def downsample(
    pcd: o3d.geometry.PointCloud,
    method: str = 'voxel',
    voxel_size: float = 0.05,
    every_k_points: int = 10,
    random_ratio: float = 0.5
) -> o3d.geometry.PointCloud:
    """
    点云降采样
    
    Args:
        pcd: 输入点云
        method: 降采样方法
            'voxel'  - 体素下采样（推荐，均匀分布）
            'uniform'- 均匀采样（每隔k个点取一个）
            'random' - 随机采样
        voxel_size: 体素大小（仅voxel方法）
        every_k_points: 采样间隔（仅uniform方法）
        random_ratio: 保留比例（仅random方法）
    
    Returns:
        降采样后的点云
    """
    logger = get_logger()
    original_count = len(pcd.points)
    
    if method == 'voxel':
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.info(f"体素降采样: voxel_size={voxel_size}")
    
    elif method == 'uniform':
        pcd_down = pcd.uniform_down_sample(every_k_points=every_k_points)
        logger.info(f"均匀降采样: every_k_points={every_k_points}")
    
    elif method == 'random':
        indices = np.random.choice(
            original_count,
            size=int(original_count * random_ratio),
            replace=False
        )
        pcd_down = pcd.select_by_index(indices)
        logger.info(f"随机降采样: ratio={random_ratio}")
    
    else:
        logger.warning(f"未知方法 {method}，返回原点云")
        return pcd
    
    new_count = len(pcd_down.points)
    logger.info(f"降采样结果: {original_count} -> {new_count} ({100*new_count/original_count:.1f}%)")
    
    return pcd_down


# ============================================================================
# 可视化
# ============================================================================

class VisConfig:
    """可视化配置"""
    def __init__(
        self,
        window_name: str = "Point Cloud Viewer",
        width: int = 1280,
        height: int = 720,
        background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        point_size: float = 2.0,
        show_coordinate_frame: bool = True,
        coordinate_frame_size: float = 1.0
    ):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.background_color = background_color
        self.point_size = point_size
        self.show_coordinate_frame = show_coordinate_frame
        self.coordinate_frame_size = coordinate_frame_size


def visualize(
    pcd: o3d.geometry.PointCloud,
    config: Optional[VisConfig] = None,
    color: Optional[Tuple[float, float, float]] = None
):
    """
    可视化点云
    
    Args:
        pcd: 点云对象
        config: 可视化配置
        color: 统一颜色(R,G,B)，范围[0,1]
    """
    logger = get_logger()
    
    if config is None:
        config = VisConfig()
    
    # 如果点云无颜色，设置默认颜色
    if not pcd.has_colors():
        if color is None:
            color = (0.6, 0.6, 0.6)
        pcd.paint_uniform_color(color)
    
    geometries = [pcd]
    
    # 添加坐标系
    if config.show_coordinate_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=config.coordinate_frame_size
        )
        geometries.append(coord_frame)
    
    logger.info(f"启动可视化: {len(pcd.points)} 个点")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=config.window_name,
        width=config.width,
        height=config.height,
        point_show_normal=False
    )


def visualize_comparison(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud,
    color1: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    color2: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    config: Optional[VisConfig] = None
):
    """
    对比可视化两个点云
    
    Args:
        pcd1, pcd2: 待对比的点云
        color1, color2: 对应颜色
        config: 可视化配置
    """
    pcd1_vis = o3d.geometry.PointCloud(pcd1)
    pcd2_vis = o3d.geometry.PointCloud(pcd2)
    
    pcd1_vis.paint_uniform_color(color1)
    pcd2_vis.paint_uniform_color(color2)
    
    merged = pcd1_vis + pcd2_vis
    visualize(merged, config)


# ============================================================================
# 工具函数
# ============================================================================

def log_pointcloud_stats(pcd: o3d.geometry.PointCloud):
    """打印点云统计信息"""
    logger = get_logger()
    
    if len(pcd.points) == 0:
        logger.warning("点云为空")
        return
    
    points = np.asarray(pcd.points)
    
    logger.info("坐标范围:")
    logger.info(f"  X: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
    logger.info(f"  Y: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
    logger.info(f"  Z: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")


def filter_by_range(
    pcd: o3d.geometry.PointCloud,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None
) -> o3d.geometry.PointCloud:
    """
    按坐标范围过滤点云
    
    Args:
        pcd: 输入点云
        x_range, y_range, z_range: (min, max) 范围，None表示不限制
    
    Returns:
        过滤后的点云
    """
    logger = get_logger()
    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    
    if x_range is not None:
        mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    if y_range is not None:
        mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    if z_range is not None:
        mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    logger.info(f"范围过滤: {len(points)} -> {len(pcd_filtered.points)}")
    
    return pcd_filtered


def filter_outliers(
    pcd: o3d.geometry.PointCloud,
    method: str = 'statistical',
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    radius: float = 0.5,
    min_points: int = 10
) -> o3d.geometry.PointCloud:
    """
    去除离群点
    
    Args:
        pcd: 输入点云
        method: 方法 ('statistical' 或 'radius')
        nb_neighbors: 邻居数量 (statistical)
        std_ratio: 标准差阈值 (statistical)
        radius: 搜索半径 (radius)
        min_points: 最小邻居数 (radius)
    
    Returns:
        去除离群点后的点云
    """
    logger = get_logger()
    original_count = len(pcd.points)
    
    if method == 'statistical':
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
    elif method == 'radius':
        pcd_clean, _ = pcd.remove_radius_outlier(
            nb_points=min_points,
            radius=radius
        )
    else:
        logger.warning(f"未知方法 {method}")
        return pcd
    
    new_count = len(pcd_clean.points)
    logger.info(f"离群点过滤({method}): {original_count} -> {new_count}")
    
    return pcd_clean


def transform_pointcloud(
    pcd: o3d.geometry.PointCloud,
    transformation: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    应用4x4变换矩阵
    
    Args:
        pcd: 输入点云
        transformation: 4x4变换矩阵
    
    Returns:
        变换后的点云
    """
    if transformation.shape != (4, 4):
        raise ValueError("变换矩阵必须是4x4")
    
    pcd_transformed = o3d.geometry.PointCloud(pcd)
    pcd_transformed.transform(transformation)
    
    get_logger().info("已应用变换矩阵")
    return pcd_transformed


def merge_pointclouds(*pcds: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """合并多个点云"""
    logger = get_logger()
    
    if len(pcds) == 0:
        return o3d.geometry.PointCloud()
    
    merged = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged += pcd
    
    logger.info(f"合并 {len(pcds)} 个点云，共 {len(merged.points)} 个点")
    return merged


# ============================================================================
# 保存
# ============================================================================

def save_pointcloud(
    pcd: o3d.geometry.PointCloud,
    output_path: str,
    binary: bool = True
) -> bool:
    """
    保存点云到文件
    
    Args:
        pcd: 点云对象
        output_path: 输出路径（支持.ply, .pcd, .xyz）
        binary: 是否二进制格式（仅ply/pcd）
    
    Returns:
        是否成功
    """
    logger = get_logger()
    
    try:
        ext = Path(output_path).suffix.lower()
        
        if ext in ['.ply', '.pcd']:
            success = o3d.io.write_point_cloud(
                output_path, pcd,
                write_ascii=not binary
            )
        elif ext in ['.xyz', '.txt']:
            np.savetxt(output_path, np.asarray(pcd.points), fmt='%.6f')
            success = True
        else:
            logger.error(f"不支持的输出格式: {ext}")
            return False
        
        if success:
            logger.info(f"已保存: {output_path} ({len(pcd.points)} 点)")
        else:
            logger.error(f"保存失败: {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"保存出错: {e}")
        return False


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 初始化日志
    logger = setup_logger(log_dir="./logs", console_output=True)
    
    # 示例：加载点云
    # pcd = load_pointcloud("example.ply")
    # pcd = load_pointcloud("data.txt", file_format='txt_bin', binary_record_format='3fi')
    # pcd = load_pointcloud("points.xyz", skip_header=1)
    
    # 示例：降采样
    # pcd_down = downsample(pcd, method='voxel', voxel_size=0.02)
    
    # 示例：过滤
    # pcd_clean = filter_outliers(pcd, method='statistical')
    # pcd_roi = filter_by_range(pcd, z_range=(0, 10))
    
    # 示例：可视化
    # visualize(pcd)
    # visualize(pcd, VisConfig(point_size=3.0, background_color=(1,1,1)))
    
    # 示例：保存
    # save_pointcloud(pcd, "output.ply", binary=True)
    
    print("点云工具模块已加载，使用 help(load_pointcloud) 查看帮助")