# Size_measuring_tools
基于三维点云尺寸测量的常用Python工具

## 项目组织
```
Size_measuring_tools/
├── 点云预处理/
│   └── PointCloud_DataLoader.py  # 核心点云处理模块
└── README.md                     # 项目说明文档
```

## 脚本文件函数列表

### PointCloud_DataLoader.py

| 函数名 | 功能描述 |
|-------|---------|
| `setup_logger()` | 配置日志记录器 |
| `get_logger()` | 获取全局日志器 |
| `set_logger()` | 设置自定义日志器 |
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

## 依赖项
- Python 3.6+
- NumPy
- Open3D
