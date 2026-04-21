import os
import numpy as np
from osgeo import gdal, osr

# 设置切片大小
TILE_SIZE = 256
# 容忍的最大云像素比例
MAX_CLOUD_PIXEL_RATIO = 0.01
MAX_CLOUD_PIXELS = TILE_SIZE * TILE_SIZE * MAX_CLOUD_PIXEL_RATIO


def process_and_slice_year(year_folder, output_base_folder):
    """
    处理特定年份文件夹下的所有TIF-QA影像对，进行切片和分类。
    结果将按切片的行列索引（R_C）组织。
    """
    print(f"--- 🚀 开始处理年份: {os.path.basename(year_folder)} ---")

    # 构造输入子文件夹路径
    tif_folder = os.path.join(year_folder, 'tif')
    qa_folder = os.path.join(year_folder, 'qa')

    if not os.path.isdir(tif_folder) or not os.path.isdir(qa_folder):
        print(f"⚠️ 错误: 找不到 {tif_folder} 或 {qa_folder}，跳过此年份。")
        return

    # 构造输出根文件夹
    year_name = os.path.basename(year_folder)
    base_cloud_dir = os.path.join(output_base_folder, year_name, 'cloud_tif')
    base_clear_dir = os.path.join(output_base_folder, year_name, 'clear_tif')

    # 无需提前创建子文件夹，因为它们将在循环中根据索引创建
    os.makedirs(base_cloud_dir, exist_ok=True)
    os.makedirs(base_clear_dir, exist_ok=True)

    # 遍历tif文件夹下的所有.tif文件
    for tif_filename in os.listdir(tif_folder):
        if not tif_filename.lower().endswith('.tif'):
            continue

        base_name = os.path.splitext(tif_filename)[0]
        tif_path = os.path.join(tif_folder, tif_filename)

        # 构造 QA 文件路径
        try:
            qa_base_name = base_name.replace('sr_band1-7', 'pixel_qa')
        except Exception:
            print(f"❌ 错误: TIF文件名 {tif_filename} 不符合预期的命名规范 'sr_band1-7'，跳过。")
            continue

        qa_filename = qa_base_name + '.tif'
        qa_path = os.path.join(qa_folder, qa_filename)

        if not os.path.exists(qa_path):
            print(f"❌ 警告: 找不到对应的QA文件: {qa_path}，跳过 {tif_filename}。")
            continue

        try:
            # 1. 打开数据
            src_ds = gdal.Open(tif_path)
            qa_ds = gdal.Open(qa_path)

            if src_ds is None or qa_ds is None:
                print(f"❌ 无法打开文件 {tif_filename} 或其QA文件。")
                continue

            cols = src_ds.RasterXSize
            rows = src_ds.RasterYSize
            bands = src_ds.RasterCount
            geotransform = src_ds.GetGeoTransform()
            proj = src_ds.GetProjection()
            driver = src_ds.GetDriver()

            # 读取整个QA数组
            qa_array = qa_ds.GetRasterBand(1).ReadAsArray().astype(np.uint16)

            # QA 逻辑：Bit 3 (值 8) 是云，Bit 5 (值 32) 是云阴影
            cloud_mask = (np.bitwise_and(qa_array, 8) == 8)
            cloud_shadow_mask = (np.bitwise_and(qa_array, 32) == 32)
            full_cloud_shadow_mask = cloud_mask | cloud_shadow_mask

            # 2. 开始切片
            tile_count = 0

            # 使用循环变量 row_index 和 col_index 来生成文件夹名称
            row_index = 0
            for j in range(0, rows, TILE_SIZE):  # j 是行（Y）的起始像素

                y_off = j
                y_size = min(TILE_SIZE, rows - j)

                # 仅处理完整的切片
                if y_size != TILE_SIZE:
                    continue

                row_index += 1
                col_index = 0
                for i in range(0, cols, TILE_SIZE):  # i 是列（X）的起始像素

                    x_off = i
                    x_size = min(TILE_SIZE, cols - i)

                    # 仅处理完整的切片
                    if x_size != TILE_SIZE:
                        continue

                    col_index += 1

                    # ----------------------------------------------------
                    # 🌟 核心修改：生成切片的行列索引文件夹名称 🌟
                    # ----------------------------------------------------
                    # 文件夹名示例：1_1, 1_2, ..., 5_5
                    tile_folder_name = f"{row_index}_{col_index}"

                    # 3. 读取切片数据
                    tile_data = src_ds.ReadAsArray(x_off, y_off, x_size, y_size)

                    # 4. QA判断逻辑
                    qa_tile_mask = full_cloud_shadow_mask[y_off: y_off + y_size,
                    x_off: x_off + x_size]

                    # 统计云像素数量
                    cloud_pixel_count = np.sum(qa_tile_mask)

                    # 5. 分类和命名
                    # 判断是 'clear_tif' 还是 'cloud_tif'，并确定基础输出目录
                    if cloud_pixel_count <= MAX_CLOUD_PIXELS:
                        base_output_dir = base_clear_dir
                    else:
                        base_output_dir = base_cloud_dir

                    # 构造最终输出路径：[base_dir]/[R_C]/[原始名称]_tile_[R]_[C].tif
                    final_output_dir = os.path.join(base_output_dir, tile_folder_name)
                    os.makedirs(final_output_dir, exist_ok=True)  # 创建 R_C 文件夹

                    # 为了在文件夹内保持文件唯一性，文件名仍包含原始信息
                    tile_name = f"{base_name}_tile_{row_index}_{col_index}.tif"
                    output_path = os.path.join(final_output_dir, tile_name)

                    # 6. 写入切片文件
                    new_geotransform = (
                        geotransform[0] + x_off * geotransform[1],
                        geotransform[1],
                        geotransform[2],
                        geotransform[3] + y_off * geotransform[5],
                        geotransform[4],
                        geotransform[5]
                    )

                    out_ds = driver.Create(output_path, x_size, y_size, bands, src_ds.GetRasterBand(1).DataType)
                    out_ds.SetGeoTransform(new_geotransform)
                    out_ds.SetProjection(proj)

                    for b in range(bands):
                        out_ds.GetRasterBand(b + 1).WriteArray(tile_data[b])

                    out_ds = None
                    tile_count += 1

            print(f"✅ 文件 {tif_filename} 处理完成，共生成 {tile_count} 个切片，并按行列索引分类存储。")

        except Exception as e:
            print(f"🛑 处理文件 {tif_filename} 时发生错误: {e}")
        finally:
            # 确保关闭GDAL数据集
            src_ds = None
            qa_ds = None


# 3. 主执行逻辑 (与之前相同)
def main(main_data_folder, output_root):
    """
    主程序，遍历所有年份文件夹。
    """
    if not os.path.exists(main_data_folder):
        print(f"主数据文件夹不存在: {main_data_folder}")
        return

    year_folders = sorted([os.path.join(main_data_folder, d)
                           for d in os.listdir(main_data_folder)
                           if os.path.isdir(os.path.join(main_data_folder, d)) and d.isdigit()])

    print(f"找到以下年份文件夹: {[os.path.basename(f) for f in year_folders]}")

    for folder in year_folders:
        process_and_slice_year(folder, output_root)

    print("--- 🎉 所有年份数据处理完成！---")

# --- 请修改以下路径以匹配您的实际情况 ---
MAIN_DATA_FOLDER = r"F:\SENMS_NEW\beijing_landsat\Beijing_1500_1500_cut_2013_2018"  # 包含 2013, 2014, ... 的父文件夹
OUTPUT_ROOT_FOLDER = r"F:\SENMS_NEW\beijing_landsat\tmp"  # 结果将保存在此文件夹下

# 执行主程序
main(MAIN_DATA_FOLDER, OUTPUT_ROOT_FOLDER)