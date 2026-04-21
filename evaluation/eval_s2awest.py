import os
import numpy as np
import rasterio
import skimage.metrics as skm
from scipy.stats import pearsonr
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def calculate_psnr_ssim(img1, img2):
    """计算单波段 PSNR 和 SSIM"""
    psnr = skm.peak_signal_noise_ratio(img1, img2, data_range=1.0)
    ssim = skm.structural_similarity(img1, img2, data_range=1.0, channel_axis=None)
    return psnr, ssim


def normalize_image(img, max_val=10000.0):
    """归一化图像到 [0,1]"""
    return img.astype(np.float32) / max_val


def save_taylor_elements_to_file(taylor_elements, save_dir):
    """保存最终每个波段的 Taylor 元素（全图像均值/整体统计）"""
    taylor_file = os.path.join(save_dir, 'taylor_elements_mean.txt')
    with open(taylor_file, 'w', encoding='utf-8') as f:
        f.write("Band,Mean_Ref,Mean_Test,Std_Ref,Std_Test,Corr_Coeff\n")
        num_bands = len(taylor_elements['mean_ref'])
        for i in range(num_bands):
            f.write(
                f"{i + 1},"
                f"{taylor_elements['mean_ref'][i]:.8f},"
                f"{taylor_elements['mean_test'][i]:.8f},"
                f"{taylor_elements['std_ref'][i]:.8f},"
                f"{taylor_elements['std_test'][i]:.8f},"
                f"{taylor_elements['corr_coeff'][i]:.8f}\n"
            )


def evaluate_images(src_dir, ref_dir, save_dir, band_mapping):
    """评估图像，并输出全数据集级别的 Taylor 元素"""
    os.makedirs(save_dir, exist_ok=True)

    band_files_ref = sorted([f for f in os.listdir(ref_dir) if f.endswith('.tif')])
    band_files_src = sorted([f for f in os.listdir(src_dir) if f.endswith('.tif')])

    if len(band_files_ref) != len(band_files_src):
        logging.warning(f"参考图像数量({len(band_files_ref)})与源图像数量({len(band_files_src)})不一致，将按 zip 后的最小数量处理。")

    # 用于累计每个波段的所有像素
    band_accumulator = {
        band_idx: {'ref': [], 'src': []}
        for band_idx in band_mapping.keys()
    }

    # 数据集级 PSNR / SSIM 统计
    psnr_values = []
    ssim_values = []

    # 如果你还想看“每个波段平均 PSNR / SSIM”
    per_band_psnr = {band_idx: [] for band_idx in band_mapping.keys()}
    per_band_ssim = {band_idx: [] for band_idx in band_mapping.keys()}

    for idx, (ref_file, src_file) in enumerate(zip(band_files_ref, band_files_src)):
        logging.info(f"正在处理文件 {idx + 1}/{min(len(band_files_ref), len(band_files_src))}: {ref_file} vs {src_file}")

        ref_path = os.path.join(ref_dir, ref_file)
        src_path = os.path.join(src_dir, src_file)

        with rasterio.open(ref_path) as ref_src:
            ref_img = normalize_image(ref_src.read())

        with rasterio.open(src_path) as src_src:
            src_img = normalize_image(src_src.read())

        for band_idx, mapped_band_idx in band_mapping.items():
            ref_band = ref_img[mapped_band_idx - 1]
            src_band = src_img[band_idx - 1]

            # 计算当前图像当前波段指标
            psnr, ssim = calculate_psnr_ssim(ref_band, src_band)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            per_band_psnr[band_idx].append(psnr)
            per_band_ssim[band_idx].append(ssim)

            # 累积像素，后续统一算 Taylor 元素
            band_accumulator[band_idx]['ref'].append(ref_band.flatten())
            band_accumulator[band_idx]['src'].append(src_band.flatten())

    # 最终每个波段只保留一个 Taylor 元素
    taylor_elements = {
        'mean_ref': [],
        'mean_test': [],
        'std_ref': [],
        'std_test': [],
        'corr_coeff': []
    }

    for band_idx in band_mapping.keys():
        ref_all = np.concatenate(band_accumulator[band_idx]['ref'], axis=0)
        src_all = np.concatenate(band_accumulator[band_idx]['src'], axis=0)

        mean_ref = np.mean(ref_all)
        mean_test = np.mean(src_all)
        std_ref = np.std(ref_all)
        std_test = np.std(src_all)
        corr_coeff, _ = pearsonr(ref_all, src_all)

        taylor_elements['mean_ref'].append(mean_ref)
        taylor_elements['mean_test'].append(mean_test)
        taylor_elements['std_ref'].append(std_ref)
        taylor_elements['std_test'].append(std_test)
        taylor_elements['corr_coeff'].append(corr_coeff)

    # 保存平均指标
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    with open(os.path.join(save_dir, "average_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Average PSNR (all images, all bands): {avg_psnr:.8f}\n")
        f.write(f"Average SSIM (all images, all bands): {avg_ssim:.8f}\n\n")

        f.write("Per-band Average PSNR / SSIM:\n")
        f.write("Band,Average_PSNR,Average_SSIM\n")
        for band_idx in band_mapping.keys():
            f.write(
                f"{band_idx},"
                f"{np.mean(per_band_psnr[band_idx]):.8f},"
                f"{np.mean(per_band_ssim[band_idx]):.8f}\n"
            )

    # 保存最终 Taylor 元素
    save_taylor_elements_to_file(taylor_elements, save_dir)

    logging.info("所有图像处理完成！")
    logging.info(f"平均 PSNR: {avg_psnr:.4f}")
    logging.info(f"平均 SSIM: {avg_ssim:.4f}")

    return taylor_elements


# =========================
# 使用示例
# =========================
src_directory = r'E:\研究生\研究生科研学习\paper3\s2awest'
ref_directory = r'M:\多时相云修复数据集\s2_asiaWest\S2PatchDataset-0117\test_GT'
save_directory = r'./s2awest_result'

# 10 波段 -> 13 波段映射
band_mapping = {
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 12,
    10: 13
}

taylor_elements = evaluate_images(src_directory, ref_directory, save_directory, band_mapping)

print("最终用于 Taylor 图的元素：")
for i in range(len(taylor_elements['mean_ref'])):
    print(
        f"Band {i+1}: "
        f"mean_ref={taylor_elements['mean_ref'][i]:.6f}, "
        f"mean_test={taylor_elements['mean_test'][i]:.6f}, "
        f"std_ref={taylor_elements['std_ref'][i]:.6f}, "
        f"std_test={taylor_elements['std_test'][i]:.6f}, "
        f"corr={taylor_elements['corr_coeff'][i]:.6f}"
    )