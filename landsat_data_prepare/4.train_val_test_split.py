import os
import shutil
import random

# =================================================================
# 配置参数
# =================================================================

# 📢 1. 请将这里替换为您上一步制作好的数据集根目录
TARGET_DATA_ROOT = r"F:\SENMS_NEW\beijing_landsat\tmp_output"

# 📢 2. 最终输出划分好的数据集的根目录
FINAL_DATASET_ROOT = r"F:\SENMS_NEW\landsat_dataset"

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# =================================================================
# 核心函数
# =================================================================

def extract_sequence_id(filename):
    """
    从文件名中提取序列ID (例如: Seq001)。
    根据文件名格式: [R]_[C]_[SeqXXX]_[T/GT]_[DATE].tif，SeqID位于索引 [2]。
    """
    # 将文件名按 '_' 分割
    parts = filename.split('_')

    # 检查长度是否足够（至少要有 R, C, SeqXXX 三部分）
    if len(parts) >= 3:
        # 序列 ID 位于索引 [2]
        seq_part = parts[2]

        # 验证这部分是否以 'Seq' 开头
        if seq_part.startswith('Seq'):
            return seq_part

    return None

def setup_output_directories(final_root):
    """创建最终的 train/val/test 及其 clear/cloudy 子目录。"""
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(final_root, split, 'clear'), exist_ok=True)
        os.makedirs(os.path.join(final_root, split, 'cloudy'), exist_ok=True)
    print("✅ 目标文件夹结构创建完成。")


def partition_dataset_by_year(target_root, final_root):
    """
    按年份遍历数据，以序列ID为单位进行 8:1:1 划分，并复制文件。
    """
    setup_output_directories(final_root)

    # 查找所有年份文件夹
    year_folders = sorted([d for d in os.listdir(target_root)
                           if os.path.isdir(os.path.join(target_root, d)) and d.isdigit()])

    if not year_folders:
        print("⚠️ 警告: 未找到年份文件夹 (YYYY) 在根目录中。")
        return

    print(f"\n--- 🚀 开始处理以下年份: {year_folders} ---")

    total_sequences_processed = 0

    for year in year_folders:
        year_path = os.path.join(target_root, year)
        print(f"\n🔄 处理年份 {year}...")

        # 1. 收集所有序列ID及其对应的所有文件路径
        sequence_map = {}  # 结构: sequence_map[seq_id] = [file_path1, file_path2, ...]

        # 遍历 clear 和 cloudy 目录
        for img_type in ['clear', 'cloudy']:
            type_path = os.path.join(year_path, img_type)
            if not os.path.isdir(type_path):
                print(f"   跳过: 找不到 {img_type} 目录。")
                continue

            for filename in os.listdir(type_path):
                if filename.lower().endswith('.tif'):
                    seq_id = extract_sequence_id(filename)
                    if seq_id:
                        file_path = os.path.join(type_path, filename)

                        if seq_id not in sequence_map:
                            sequence_map[seq_id] = []
                        sequence_map[seq_id].append(file_path)

        if not sequence_map:
            print(f"   警告: 年份 {year} 中未找到有效的序列数据。")
            continue

        all_sequences = list(sequence_map.keys())
        random.shuffle(all_sequences)  # 随机打乱序列，确保划分随机性

        num_sequences = len(all_sequences)
        print(f"   找到 {num_sequences} 个独立序列，开始 8:1:1 划分。")

        # 2. 计算划分索引
        num_train = int(num_sequences * TRAIN_RATIO)
        num_val = int(num_sequences * VAL_RATIO)
        # 剩余的都是测试集 (确保总数准确)
        num_test = num_sequences - num_train - num_val

        # 3. 执行划分
        train_sequences = all_sequences[:num_train]
        val_sequences = all_sequences[num_train: num_train + num_val]
        test_sequences = all_sequences[num_train + num_val:]

        partition = {
            'train': train_sequences,
            'val': val_sequences,
            'test': test_sequences
        }

        print(f"   划分结果: Train({len(train_sequences)}) / Val({len(val_sequences)}) / Test({len(test_sequences)})")

        # 4. 复制文件到最终目录
        for split_name, seq_list in partition.items():

            for seq_id in seq_list:
                file_paths = sequence_map[seq_id]
                total_sequences_processed += 1

                for src_path in file_paths:
                    filename = os.path.basename(src_path)

                    # 判断是 clear 还是 cloudy 文件
                    # T/GT 标签在文件名中，但更准确的是从原始路径判断
                    # 从 filename 中判断：如果包含 'GT' 则是 clear，否则是 cloudy
                    if '_GT_' in filename:
                        img_type = 'clear'
                    else:
                        img_type = 'cloudy'

                    # 构造目标路径: /FINAL_DATASET_ROOT/split_name/img_type/filename.tif
                    dest_dir = os.path.join(final_root, split_name, img_type)
                    dest_path = os.path.join(dest_dir, filename)

                    shutil.copy2(src_path, dest_path)

    print(f"\n--- 🎉 所有年份数据处理完毕 ---")
    print(f"总共处理并复制了 {total_sequences_processed} 个序列的所有文件。")

# --- 运行主程序 ---
partition_dataset_by_year(TARGET_DATA_ROOT, FINAL_DATASET_ROOT)