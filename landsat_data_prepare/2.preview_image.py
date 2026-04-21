import os
import numpy as np
from osgeo import gdal
from PIL import Image

# =================================================================
# 配置参数
# =================================================================

# Landsat 8/9 的波段索引（基于 GDAL/Python 数组的 0-based 索引）
# Band 4 (Red) -> 索引 3
# Band 3 (Green) -> 索引 2
# Band 2 (Blue) -> 索引 1
R_BAND_INDEX = 3
G_BAND_INDEX = 2
B_BAND_INDEX = 1

# 默认拉伸参数：2% 线性截断拉伸
LOWER_PERCENT = 2
UPPER_PERCENT = 98


# =================================================================
# 核心函数
# =================================================================

def normalize_and_convert_to_8bit(band_array):
    """
    对单个波段数组进行线性 2% 截断拉伸，并转换为 8 位 (0-255) 数组。
    """
    # 扁平化数组，用于计算百分位数，同时移除可能存在的 NoData 值（例如 0）
    valid_data = band_array[band_array > 0]

    if valid_data.size == 0:
        return np.zeros_like(band_array, dtype=np.uint8)

    # 1. 计算拉伸范围（百分位截断）
    min_val = np.percentile(valid_data, LOWER_PERCENT)
    max_val = np.percentile(valid_data, UPPER_PERCENT)

    # 避免除以零或拉伸范围过小
    if max_val <= min_val:
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val <= min_val:
            return np.zeros_like(band_array, dtype=np.uint8)

    # 2. 线性拉伸并截断
    stretched_array = np.clip(band_array, min_val, max_val)

    # 3. 归一化到 0-1
    normalized_array = (stretched_array - min_val) / (max_val - min_val)

    # 4. 转换为 8 位 (0-255)
    output_8bit = (normalized_array * 255).astype(np.uint8)

    return output_8bit


def create_true_color_png(tif_path, output_png_path):
    """
    打开TIF文件，提取Band 4, 3, 2，生成真彩色PNG。
    """
    try:
        src_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
        if src_ds is None:
            print(f"❌ 错误: 无法打开文件 {tif_path}")
            return False

        # 检查波段数量是否足够
        if src_ds.RasterCount <= max(R_BAND_INDEX, G_BAND_INDEX, B_BAND_INDEX):
            print(f"🛑 警告: 文件 {tif_path} 波段数量 ({src_ds.RasterCount}) 不足 4 个。")
            src_ds = None
            return False

        # 1. 读取波段数据 (GDAL Band索引从1开始)
        R_band_data = src_ds.GetRasterBand(R_BAND_INDEX + 1).ReadAsArray()
        G_band_data = src_ds.GetRasterBand(G_BAND_INDEX + 1).ReadAsArray()
        B_band_data = src_ds.GetRasterBand(B_BAND_INDEX + 1).ReadAsArray()

        src_ds = None  # 关闭文件

        # 2. 归一化和转换为 8 位
        R_8bit = normalize_and_convert_to_8bit(R_band_data)
        G_8bit = normalize_and_convert_to_8bit(G_band_data)
        B_8bit = normalize_and_convert_to_8bit(B_band_data)

        # 3. 合并成 RGB 图像 (Pillow 需要 (rows, cols, 3) 顺序)
        rgb_stack = np.dstack((R_8bit, G_8bit, B_8bit))

        # 4. 创建 Pillow 图像对象并保存
        img = Image.fromarray(rgb_stack, 'RGB')

        # 确保输出目录存在 (在 batch_convert_tifs_to_png 中已确认)
        img.save(output_png_path)
        return True

    except Exception as e:
        print(f"🛑 处理 {tif_path} 时发生异常: {e}")
        return False


def batch_convert_tifs_to_png_in_place(root_dir):
    """
    递归遍历根目录下的所有 .tif 文件，并将对应的 .png 文件保存到同一目录下。
    """
    processed_count = 0

    print(f"--- 🔍 开始遍历目录: {root_dir} ---")

    # 使用 os.walk 递归遍历所有子目录
    for dirpath, _, filenames in os.walk(root_dir):

        for filename in filenames:
            if filename.lower().endswith('.tif'):
                tif_path = os.path.join(dirpath, filename)

                # 构造 PNG 输出路径：与 TIF 文件在同一目录
                base_name = os.path.splitext(filename)[0]
                png_filename = base_name + '.png'
                output_png_path = os.path.join(dirpath, png_filename)  # 直接使用 dirpath

                # 打印相对路径以跟踪进度
                relative_path_display = os.path.relpath(tif_path, root_dir)
                print(f"-> 正在处理: {relative_path_display}")

                if create_true_color_png(tif_path, output_png_path):
                    processed_count += 1

    print(f"--- 🎉 批量转换完成。总共成功处理 {processed_count} 个文件。 ---")


# =================================================================
# 运行示例
# =================================================================

# 📢 请根据您的实际路径修改以下变量！
# 包含您的 2013/clear_tif/1_1 等切片文件夹的根目录
INPUT_DATA_ROOT = r"F:\SENMS_NEW\landsat_dataset"

# # 执行批量转换
batch_convert_tifs_to_png_in_place(INPUT_DATA_ROOT)