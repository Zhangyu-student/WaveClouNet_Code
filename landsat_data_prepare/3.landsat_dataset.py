import os
import re
import shutil
from datetime import datetime
import numpy as np

# =================================================================
# 配置参数
# =================================================================

# 📢 1. 请将这里替换为您第一步切片输出的根目录
INPUT_DATA_ROOT = r"F:\SENMS_NEW\beijing_landsat\tmp"

# 📢 2. 请将这里替换为您希望存放最终匹配结果的根目录
TARGET_DATA_ROOT = r"F:\SENMS_NEW\beijing_landsat\tmp_output"

# 文件名中的日期格式 (e.g., 20150416)
DATE_FORMAT = "%Y%m%d"

# 正则表达式用于从文件名中提取日期和切片位置
# 文件名示例: LC08_L1TP_123032_20150416_20170409_01_T1_sr_band1-7mosaic_clip_tile_1_3.tif
FILENAME_PATTERN = re.compile(r'_(\d{8})_.*?_tile_(\d+)_(\d+)\.tif$')


# =================================================================
# 辅助类：存储文件元数据
# =================================================================

class ImageMetadata:
    def __init__(self, full_path, year, date, tile_rc):
        self.full_path = full_path
        self.year = year
        self.date = date  # datetime object
        self.tile_rc = tile_rc  # "R_C" string
        self.filename = os.path.basename(full_path)


# =================================================================
# 核心函数：文件解析与分组
# =================================================================

def parse_filename(full_path):
    """从文件路径和名称中提取年份、日期和切片位置。"""
    filename = os.path.basename(full_path)
    match = FILENAME_PATTERN.search(filename)
    if not match: return None
    date_str = match.group(1)
    tile_r = match.group(2)
    tile_c = match.group(3)
    tile_rc = f"{tile_r}_{tile_c}"

    # 路径结构预期: INPUT_DATA_ROOT/2013/cloud_tif/1_1/file.tif
    try:
        # 获取 /2013/ 部分作为年份
        year = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
        if not year.isdigit(): return None
        year = int(year)
    except:
        return None

    try:
        date_obj = datetime.strptime(date_str, DATE_FORMAT)
    except ValueError:
        return None
    return ImageMetadata(full_path, year, date_obj, tile_rc)


def load_and_group_files(input_root):
    """遍历输入目录，加载所有TIF文件并按 (年份, 切片位置) 分组。"""
    print("\n--- 1. 正在加载和分组切片文件 ---")
    data = {}  # 结构: data[year][tile_rc] = {'cloud': [Metadata], 'clear': [Metadata]}

    for dirpath, _, filenames in os.walk(input_root):
        if 'cloud_tif' in dirpath:
            img_type = 'cloud'
        elif 'clear_tif' in dirpath:
            img_type = 'clear'
        else:
            continue

        for filename in filenames:
            if not filename.lower().endswith('.tif'): continue
            full_path = os.path.join(dirpath, filename)
            metadata = parse_filename(full_path)

            if metadata:
                year_key = metadata.year
                rc_key = metadata.tile_rc
                if year_key not in data: data[year_key] = {}
                if rc_key not in data[year_key]: data[year_key][rc_key] = {'cloud': [], 'clear': []}
                data[year_key][rc_key][img_type].append(metadata)

    # 按日期排序
    for year in data:
        for rc in data[year]:
            data[year][rc]['cloud'].sort(key=lambda x: x.date)
            data[year][rc]['clear'].sort(key=lambda x: x.date)

    print(f"   文件加载完成，共找到 {len(data)} 个年份的数据。")
    return data


def find_nearest_clear_gt(seq_date, clear_list):
    """从清晰影像列表中，找到日期最接近序列中心日期 (seq_date) 的影像。"""
    if not clear_list: return None
    min_diff = float('inf')
    nearest_clear = None
    for clear_meta in clear_list:
        time_diff = abs((clear_meta.date - seq_date).days)
        if time_diff < min_diff:
            min_diff = time_diff
            nearest_clear = clear_meta
    return nearest_clear


# =================================================================
# 核心函数：时序匹配与数据复制
# =================================================================

def match_and_copy_sequences(data, target_root):
    """
    进行 3-时相云影像与 1-真值清晰影像的匹配，使用单张交叉重叠序列提取 (123, 345, 567...)。
    并使用简化的命名和目录结构进行复制。
    """
    print("\n--- 2. 正在匹配 3-时相序列和真值影像 (单张交叉重叠) ---")
    global_sequence_id = 0

    for year, rc_data in data.items():
        # 构造目标年文件夹下的 Cloudy 和 Clear 目录
        cloudy_dest_dir = os.path.join(target_root, str(year), 'cloudy')
        clear_dest_dir = os.path.join(target_root, str(year), 'clear')
        os.makedirs(cloudy_dest_dir, exist_ok=True)
        os.makedirs(clear_dest_dir, exist_ok=True)

        for rc, file_lists in rc_data.items():
            cloud_list = file_lists['cloud']
            clear_list = file_lists['clear']

            # 序列提取：步长为 2 实现单张交叉重叠 (123, 345, 567...)
            sequences = []
            for i in range(0, len(cloud_list) - 2, 2):
                sequences.append([cloud_list[i], cloud_list[i + 1], cloud_list[i + 2]])

            if not sequences:
                continue

            for cloud_sequence in sequences:
                global_sequence_id += 1

                # 序列中心日期（使用中间那张影像的日期）
                seq_date = cloud_sequence[1].date
                seq_id_str = f"Seq{global_sequence_id:03d}"

                clear_gt = find_nearest_clear_gt(seq_date, clear_list)

                if not clear_gt:
                    print(f"警告: 在 {year}/{rc} 中找不到清晰真值，跳过序列 {global_sequence_id}。")
                    continue

                # 复制 3 张云影像
                for t_index, meta in enumerate(cloud_sequence):
                    t_label = f"T{t_index + 1}"  # T1, T2, T3
                    # 新文件名: [切片R_C]_[序列ID]_[时序T1/T2/T3]_[原日期].tif
                    new_filename = f"{rc}_{seq_id_str}_{t_label}_{meta.date.strftime(DATE_FORMAT)}.tif"
                    target_path = os.path.join(cloudy_dest_dir, new_filename)
                    shutil.copy2(meta.full_path, target_path)

                # 复制 1 张清晰影像 (真值)
                # 新文件名: [切片R_C]_[序列ID]_GT_[原日期].tif
                clear_filename = f"{rc}_{seq_id_str}_GT_{clear_gt.date.strftime(DATE_FORMAT)}.tif"
                target_path = os.path.join(clear_dest_dir, clear_filename)
                shutil.copy2(clear_gt.full_path, target_path)

    print(f"   文件复制完成。总共生成 {global_sequence_id} 个匹配序列 (共 {global_sequence_id * 4} 个文件)。")


# =================================================================
# 主函数
# =================================================================

def main_data_processing():
    """
    执行数据集制作的第二步：时序筛选和数据组织。
    """
    if not os.path.isdir(INPUT_DATA_ROOT):
        print(f"错误: 输入数据目录不存在: {INPUT_DATA_ROOT}")
        print("请检查 INPUT_DATA_ROOT 变量是否已正确设置为您切片后的数据路径。")
        return

    os.makedirs(TARGET_DATA_ROOT, exist_ok=True)

    # 步骤 1: 加载并分组文件
    file_data = load_and_group_files(INPUT_DATA_ROOT)

    # 步骤 2: 匹配并复制到目标目录
    match_and_copy_sequences(file_data, TARGET_DATA_ROOT)

    print("\n--- 🎉 数据集制作的第二步（时序筛选）已完成！ ---")

# 运行主程序
main_data_processing()