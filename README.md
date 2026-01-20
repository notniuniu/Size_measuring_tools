# Size_measuring_tools
基于三维点云尺寸测量的常用Python工具

## 项目组织
```
Size_measuring_tools/
├── 点云预处理/
│   ├── PointCloud_DataLoader.py  # 核心点云处理模块
│   ├── Depth_map_renderin.py     # 深度图渲染模块
│   ├── ICP_registration.py       # 点云配准模块
│   ├── adaptive_downsample.py    # 自适应降采样模块
│   └── pointcloud_denoise.py     # 点云去噪模块
├── 点云分割/
│   └── extract_plane.py          # 平面提取模块
└── README.md                     # 项目说明文档
```

## 脚本文件函数列表

### PointCloud_DataLoader.py

| 函数名 | 功能描述 |
|-------|---------|
| `load_pointcloud()` | 统一点云加载接口，自动识别格式 |
| `downsample()` | 点云降采样（支持体素、均匀、随机三种方法） |
| `visualize()` | 可视化点云 |
| `visualize_comparison()` | 对比可视化两个点云 |
| `log_pointcloud_stats()` | 打印点云统计信息 |
| `filter_by_range()` | 按坐标范围过滤点云 |
| `filter_outliers()` | 去除离群点（支持统计和半径两种方法） |
| `transform_pointcloud()` | 应用4x4变换矩阵 |
| `merge_pointclouds()` | 合并多个点云 |
| `save_pointcloud()` | 保存点云到文件 |

### Depth_map_renderin.py

| 函数名 | 功能描述 |
|-------|---------|
| `generate_depth_map()` | 从点云数据生成深度图，支持遮挡处理 |
| `otsu_threshold_improved()` | 改进的Otsu阈值算法，用于自动分割前景背景 |
| `depth_map_to_pointcloud()` | 将深度图逆变换回点云格式 |
| `compute_depth_statistics()` | 计算深度图的统计信息 |
| `batch_generate_depth_maps()` | 批量生成多个点云的深度图 |

### ICP_registration.py

| 函数名 | 功能描述 |
|-------|---------|
| `icp_registration()` | ICP点云配准，实现核心配准算法 |
| `extract_overlap_region()` | 提取点云重叠区域（基于Z轴阈值分割） |
| `batch_registration()` | 批量点云配准（源与目标一一对应） |
| `evaluate_registration()` | 评估配准质量 |
| `create_transform_matrix()` | 创建4×4变换矩阵 |
| `decompose_transform()` | 分解变换矩阵为旋转和平移 |
| `merge_registered_pointclouds()` | 合并配准后的点云 |

### adaptive_downsample.py

| 函数名 | 功能描述 |
|-------|---------|
| `compute_bounding_box()` | 计算点云包围盒信息 |
| `estimate_point_density()` | 估计点云密度（点数/体积） |
| `compute_adaptive_voxel_size()` | 根据点数规模自适应计算体素尺寸 |
| `adaptive_voxel_downsample()` | 核心自适应体素降采样函数 |
| `batch_adaptive_downsample()` | 批量自适应降采样 |
| `uniform_downsample()` | 均匀降采样（每隔k个点取一个） |
| `random_downsample()` | 随机降采样 |
| `farthest_point_downsample()` | 最远点采样（FPS） |
| `evaluate_downsample_quality()` | 评估降采样质量 |

### pointcloud_denoise.py

| 函数名 | 功能描述 |
|-------|---------|
| `dbscan_denoise()` | 基于DBSCAN密度聚类的点云去噪 |
| `statistical_denoise()` | 统计滤波去噪 |
| `radius_denoise()` | 半径滤波去噪 |
| `pipeline_denoise()` | 组合去噪流水线 |
| `estimate_dbscan_params()` | 自适应估计DBSCAN参数 |
| `batch_denoise()` | 批量点云去噪 |
| `evaluate_denoise_quality()` | 评估去噪质量 |

### extract_plane.py

| 函数名 | 功能描述 |
|-------|---------|
| `compute_bounding_box()` | 计算点云包围盒 |
| `get_reference_point()` | 根据比例参考点获取实际空间位置 |
| `get_local_region()` | 获取局部搜索区域 |
| `compute_plane_from_points()` | 从三个点计算平面方程 |
| `get_point_plane_distance()` | 计算点到平面的距离 |
| `check_normal_consistency()` | 检查两个法向量是否一致 |
| `ransac_plane_extraction()` | 使用RANSAC方法提取平面 |
| `region_growing_plane_extraction()` | 使用区域增长方法提取平面 |
| `extract_plane()` | 面提取主函数 |

## 依赖项
- Python 3.6+
- NumPy
- Open3D
- OpenCV (cv2)
- SciPy
