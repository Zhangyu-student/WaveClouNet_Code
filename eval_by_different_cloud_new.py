# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
from torch.utils.data import Dataset, DataLoader
from lpips import LPIPS
from cleanfid import fid
from tqdm import tqdm
import json
import time
import re
import shutil


# ===========================
#   Cloud map helpers
# ===========================

# 新命名：T12TUR_R027_0
NEW_TILE_RE = re.compile(r"^(T\d{2}[A-Z]{3}_R\d{3})_\d+$", re.IGNORECASE)

# 旧命名：01WFN_60009000
OLD_REGION_RE = re.compile(r"(\d{2}[A-Z]{3}_[0-9]+)", re.IGNORECASE)


def extract_tile_id(name: str):
    """从 T12TUR_R027_0 提取 tile: T12TUR_R027"""
    m = NEW_TILE_RE.match(name.strip())
    return m.group(1).upper() if m else None


def load_cloud_map(txt_path: str):
    """
    读取云量/污染面积txt：每行至少前两列为 'key value' (tab/空格都可)，后续列会被忽略
    例如：
      - T12TUR_R027_0    63.4281
      - T12TUR_R027_0    63.4281    60+
    key 可能是：
      - 01WFN_60009000
      - T12TUR_R027_3
      - T12TUR_R027
      - T12TUR_R027_0
    """
    cloud = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            key = parts[0].strip().upper()
            try:
                val = float(parts[1])
            except:
                continue
            cloud[key] = val
    return cloud


def build_tile_mean_cloud(cloud_map: dict):
    """
    由 cloud_map 构建 tile -> mean_cloud
    支持：
      - key = T12TUR_R027_3 （patch）
      - key = T12TUR_R027   （tile）
    如果 tile 本身有值，就优先使用 tile 值；
    否则用该 tile 下所有 patch 的平均值。
    """
    tile_to_vals = {}
    tile_direct = {}

    for k, v in cloud_map.items():
        k = k.upper()

        # 直接 tile
        if re.match(r"^T\d{2}[A-Z]{3}_R\d{3}$", k, re.IGNORECASE):
            tile_direct[k] = float(v)
            continue

        # patch -> tile
        tile = extract_tile_id(k)
        if tile is not None:
            tile_to_vals.setdefault(tile, []).append(float(v))

    tile_mean = {}
    # 优先使用 tile 直接值
    for tile, v in tile_direct.items():
        tile_mean[tile] = float(v)

    # 没有直接值的，再用 patch 均值补
    for tile, vals in tile_to_vals.items():
        if tile not in tile_mean and len(vals) > 0:
            tile_mean[tile] = float(np.mean(vals))

    return tile_mean


def get_cloud_value_for_filename(filename_no_ext: str, cloud_map: dict, tile_mean_map: dict):
    """
    给定评估 filename（如 T12TUR_R027_0），返回云量值：
    1) 先用完整 key: T12TUR_R027_0（如果txt里有）
    2) 再用 tile key: T12TUR_R027（如果txt里有）
    3) 再用 tile_mean_map[tile]（从 txt 的同 tile 所有 patch 平均得到）
    4) 兼容旧命名：01WFN_60009000（如果 filename 里含这种）
    """
    name = filename_no_ext.strip().upper()

    # 1) 完整 key
    if name in cloud_map:
        return float(cloud_map[name])

    # 2) tile key
    tile = extract_tile_id(name)
    if tile is not None:
        if tile in cloud_map:
            return float(cloud_map[tile])
        if tile in tile_mean_map:
            return float(tile_mean_map[tile])

    # 3) 旧命名兼容（如果 filename 中含 01WFN_60009000）
    m = OLD_REGION_RE.search(name)
    if m:
        k = m.group(1).upper()
        if k in cloud_map:
            return float(cloud_map[k])

    return None


def assign_cloud_bin(val: float, bins, labels):
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        if (val >= left) and (val < right or (i == len(bins) - 2 and val <= right)):
            return labels[i]
    return None


def build_cloud_labels(bins):
    labels = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        if np.isinf(right):
            labels.append(f"{left:g}+")
        else:
            labels.append(f"{left:g}-{right:g}")
    return labels


# ===========================
#   Metrics helpers
# ===========================

def calculate_sam_rgb(gt_rgb, pred_rgb):
    if gt_rgb.shape[-1] != 3 or pred_rgb.shape[-1] != 3:
        raise ValueError("Input images must have 3 channels (RGB)")

    gt_rgb = gt_rgb.astype(np.float32)
    pred_rgb = pred_rgb.astype(np.float32)

    gt_flat = gt_rgb.reshape(-1, 3)
    pred_flat = pred_rgb.reshape(-1, 3)

    dot_product = np.sum(gt_flat * pred_flat, axis=1)
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)

    valid_mask = (norm_gt > 1e-6) & (norm_pred > 1e-6)
    dot_product = dot_product[valid_mask]
    norm_gt = norm_gt[valid_mask]
    norm_pred = norm_pred[valid_mask]

    cos_theta = dot_product / (norm_gt * norm_pred)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return np.mean(angle_deg)


def calculate_lpips(lpips_model, gt_img, pred_img, device):
    def numpy_to_torch(img):
        img_tensor = torch.from_numpy(img.astype(np.float32)) / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        return img_tensor

    gt_tensor = numpy_to_torch(gt_img)
    pred_tensor = numpy_to_torch(pred_img)

    with torch.no_grad():
        lpips_val = lpips_model(gt_tensor, pred_tensor).item()

    return lpips_val


def save_images_for_fid(images, filenames, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for img, filename in zip(images, filenames):
        base_name = os.path.splitext(filename)[0]
        Image.fromarray(img).save(os.path.join(save_dir, f"{base_name}.png"))


def list_image_files(folder):
    return (
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.jpeg"))
    )


# ===========================
#   Dataset (严格同名匹配!)
# ===========================

class ImagePairDataset(Dataset):
    """严格同名匹配：pred_stem 必须等于 gt_stem，避免 _1 匹配到 _10"""

    def __init__(self, gt_dir, pred_dir):
        gt_files = list_image_files(gt_dir)
        pred_files = list_image_files(pred_dir)

        gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_files}

        self.gt_paths = []
        self.pred_paths = []
        self.filenames = []

        for pred_path in sorted(pred_files):
            pred_name = os.path.splitext(os.path.basename(pred_path))[0]
            gt_path = gt_map.get(pred_name, None)
            if gt_path is not None:
                self.gt_paths.append(gt_path)
                self.pred_paths.append(pred_path)
                self.filenames.append(pred_name)

        print(f"[Pairing] {os.path.basename(pred_dir)}: 找到 {len(self.filenames)} 对匹配图像（严格同名）")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        try:
            gt_img = np.array(Image.open(self.gt_paths[idx]).convert("RGB"))
            pred_img = np.array(Image.open(self.pred_paths[idx]).convert("RGB"))
        except Exception as e:
            print(f"读取图像 {self.filenames[idx]} 时出错: {str(e)}")
            gt_img = np.zeros((256, 256, 3), dtype=np.uint8)
            pred_img = np.zeros((256, 256, 3), dtype=np.uint8)

        return {"gt": gt_img, "pred": pred_img, "filename": self.filenames[idx]}


# ===========================
#   Evaluate one method
# ===========================

def evaluate_one(
    gt_dir, pred_dir, output_dir,
    lpips_net="alex", workers=4, use_cpu=False,
    cloud_stats=False, cloud_map=None, tile_mean_map=None, cloud_bins=None
):
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "gt_dir": gt_dir,
        "pred_dir": pred_dir,
        "output_dir": output_dir,
        "lpips_net": lpips_net,
        "workers": workers,
        "use_cpu": use_cpu,
        "cloud_stats": cloud_stats,
        "cloud_bins": cloud_bins,
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n========== Evaluating: {os.path.basename(pred_dir)} ==========")
    print(f"device: {device}")

    lpips_model = LPIPS(net=lpips_net).to(device)
    lpips_model.eval()

    dataset = ImagePairDataset(gt_dir, pred_dir)
    if len(dataset) == 0:
        print(f"[WARN] {os.path.basename(pred_dir)}: 没有找到匹配图像对，跳过。")
        return None

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)

    metrics = {"psnr": [], "ssim": [], "sam": [], "lpips": [], "filenames": dataset.filenames}

    # FID temp dirs
    fid_real_dir = os.path.join(output_dir, "fid_real")
    fid_pred_dir = os.path.join(output_dir, "fid_pred")
    for d in [fid_real_dir, fid_pred_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    fid_real_imgs, fid_pred_imgs, fid_filenames = [], [], []

    # 云量分桶
    bins = cloud_bins if cloud_bins is not None else [0.0, 20.0, 40.0, 60.0, float("inf")]
    labels = build_cloud_labels(bins)
    bin_metrics = {lb: {"psnr": [], "ssim": [], "sam": [], "lpips": [], "n": 0} for lb in labels}
    missing_cloud = []

    for batch in tqdm(dataloader, desc=f"{os.path.basename(pred_dir)}"):
        gt_img = batch["gt"][0].numpy()
        pred_img = batch["pred"][0].numpy()
        filename = batch["filename"][0]

        fid_real_imgs.append(gt_img)
        fid_pred_imgs.append(pred_img)
        fid_filenames.append(filename)

        if gt_img.shape[:2] != pred_img.shape[:2]:
            min_h = min(gt_img.shape[0], pred_img.shape[0])
            min_w = min(gt_img.shape[1], pred_img.shape[1])
            gt_img = gt_img[:min_h, :min_w]
            pred_img = pred_img[:min_h, :min_w]

        try:
            psnr_val = psnr(gt_img, pred_img, data_range=255)

            # 更稳健的写法（避免 multichannel 参数在不同版本行为差异）
            ssim_val = ssim(
                gt_img, pred_img,
                channel_axis=-1,
                data_range=255,
                gaussian_weights=True,
                use_sample_covariance=False,
                sigma=1.5
            )

            sam_val = calculate_sam_rgb(gt_img, pred_img)
            lpips_val = calculate_lpips(lpips_model, gt_img, pred_img, device)
        except Exception as e:
            print(f"计算 {filename} 的指标时出错: {str(e)}")
            psnr_val = ssim_val = sam_val = lpips_val = float('nan')

        metrics["psnr"].append(psnr_val)
        metrics["ssim"].append(ssim_val)
        metrics["sam"].append(sam_val)
        metrics["lpips"].append(lpips_val)

        # 按云量分桶（可选）
        if cloud_stats and cloud_map is not None and tile_mean_map is not None and (not np.isnan(psnr_val)):
            cval = get_cloud_value_for_filename(filename, cloud_map, tile_mean_map)
            if cval is None:
                missing_cloud.append(filename)
            else:
                lb = assign_cloud_bin(float(cval), bins=bins, labels=labels)
                if lb is not None:
                    bin_metrics[lb]["psnr"].append(psnr_val)
                    bin_metrics[lb]["ssim"].append(ssim_val)
                    bin_metrics[lb]["sam"].append(sam_val)
                    bin_metrics[lb]["lpips"].append(lpips_val)
                    bin_metrics[lb]["n"] += 1

    # FID
    save_images_for_fid(fid_real_imgs, fid_filenames, fid_real_dir)
    save_images_for_fid(fid_pred_imgs, fid_filenames, fid_pred_dir)

    try:
        fid_value = fid.compute_fid(fid_real_dir, fid_pred_dir, device=device, num_workers=0)
    except Exception as e:
        print(f"计算FID时出错: {str(e)}")
        fid_value = float('nan')

    avg_metrics = {
        "psnr": float(np.nanmean(metrics["psnr"])),
        "ssim": float(np.nanmean(metrics["ssim"])),
        "sam": float(np.nanmean(metrics["sam"])),
        "lpips": float(np.nanmean(metrics["lpips"])),
        "fid": float(fid_value)
    }

    # save detailed
    detail_path = os.path.join(output_dir, "detailed_metrics.csv")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write("filename,psnr,ssim,sam,lpips\n")
        for i, fn in enumerate(metrics["filenames"]):
            f.write(f"{fn},{metrics['psnr'][i]:.6f},{metrics['ssim'][i]:.6f},{metrics['sam'][i]:.6f},{metrics['lpips'][i]:.6f}\n")

    # save summary
    summary_path = os.path.join(output_dir, "summary_metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("图像质量评估结果汇总\n")
        f.write("================================\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"真实图像目录: {gt_dir}\n")
        f.write(f"预测图像目录: {pred_dir}\n")
        f.write(f"总图像数: {len(dataset)}\n\n")
        f.write(f"平均 PSNR: {avg_metrics['psnr']:.6f} dB\n")
        f.write(f"平均 SSIM: {avg_metrics['ssim']:.6f}\n")
        f.write(f"平均 SAM:  {avg_metrics['sam']:.6f}°\n")
        f.write(f"平均 LPIPS:{avg_metrics['lpips']:.6f}\n")
        f.write(f"FID:       {avg_metrics['fid']:.6f}\n\n")

    # save by-cloud summary
    if cloud_stats:
        bycloud_path = os.path.join(output_dir, "summary_metrics_by_cloud.txt")
        with open(bycloud_path, "w", encoding="utf-8") as f:
            f.write("按云量分段的图像质量评估汇总\n")
            f.write("================================\n")
            f.write(f"bins: {bins}\n\n")
            for lb in labels:
                n = bin_metrics[lb]["n"]
                if n == 0:
                    f.write(f"[{lb}] n=0\n\n")
                    continue
                f.write(f"[{lb}] n={n}\n")
                f.write(f"  PSNR : {float(np.mean(bin_metrics[lb]['psnr'])):.6f}\n")
                f.write(f"  SSIM : {float(np.mean(bin_metrics[lb]['ssim'])):.6f}\n")
                f.write(f"  SAM  : {float(np.mean(bin_metrics[lb]['sam'])):.6f}\n")
                f.write(f"  LPIPS: {float(np.mean(bin_metrics[lb]['lpips'])):.6f}\n\n")

            f.write(f"未能匹配云量的样本数: {len(missing_cloud)}\n")
            for x in missing_cloud[:50]:
                f.write(f"  {x}\n")

    print(f"[Done] {os.path.basename(pred_dir)} -> {output_dir}")
    return avg_metrics


# ===========================
#   Evaluate multiple methods
# ===========================

def collect_method_dirs(pred_root, methods=None):
    """
    pred_root 下面每个子文件夹视为一个算法方法（WaveCloudNet、PMAA...）
    - methods=None: 自动遍历所有子文件夹
    - methods=[...]: 只跑指定方法名（子文件夹名）
    """
    pred_root = os.path.abspath(pred_root)
    if not os.path.isdir(pred_root):
        raise FileNotFoundError(f"pred_root not found: {pred_root}")

    all_subdirs = [d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))]
    all_subdirs.sort()

    if methods is None:
        picked = all_subdirs
    else:
        methods_set = set([m.strip() for m in methods])
        picked = [d for d in all_subdirs if d in methods_set]

    return [os.path.join(pred_root, d) for d in picked]


def evaluate_multi(
    gt_dir,
    pred_root=None,
    pred_dirs=None,
    output_root="./evaluation_results_multi",
    lpips_net="alex",
    workers=4,
    use_cpu=False,
    cloud_stats=False,
    cloud_txt_path=None,
    cloud_bins=None,
    methods=None,
):
    """
    支持两种方式：
    A) pred_root=...（postprocess 目录），自动扫描其子文件夹作为多个算法
    B) pred_dirs=[path1, path2, ...] 手动指定多个算法输出目录
    """
    os.makedirs(output_root, exist_ok=True)

    # 云量映射只加载一次
    cloud_map = tile_mean_map = None
    if cloud_stats:
        if (cloud_txt_path is None) or (not os.path.exists(cloud_txt_path)):
            raise FileNotFoundError("cloud_stats=True 但 cloud_txt_path 不存在/无效")
        cloud_map = load_cloud_map(cloud_txt_path)
        tile_mean_map = build_tile_mean_cloud(cloud_map)
        print(f"[Cloud] loaded keys={len(cloud_map)}, tiles={len(tile_mean_map)} from {cloud_txt_path}")

    # 组织 pred_dirs
    if pred_dirs is None:
        if pred_root is None:
            raise ValueError("请提供 pred_root 或 pred_dirs")
        pred_dirs = collect_method_dirs(pred_root, methods=methods)

    # 跑每个方法
    all_results = []
    for pd in pred_dirs:
        method_name = os.path.basename(pd.rstrip("/\\"))
        out_dir = os.path.join(output_root, method_name)

        avg = evaluate_one(
            gt_dir=gt_dir,
            pred_dir=pd,
            output_dir=out_dir,
            lpips_net=lpips_net,
            workers=workers,
            use_cpu=use_cpu,
            cloud_stats=cloud_stats,
            cloud_map=cloud_map,
            tile_mean_map=tile_mean_map,
            cloud_bins=cloud_bins
        )
        if avg is not None:
            all_results.append((method_name, avg))

    # 写一个总表方便你看
    summary_all = os.path.join(output_root, "summary_all_methods.txt")
    with open(summary_all, "w", encoding="utf-8") as f:
        f.write("All methods summary\n")
        f.write("================================\n")
        f.write(f"GT: {gt_dir}\n")
        if pred_root is not None:
            f.write(f"Pred root: {pred_root}\n")
        f.write(f"Methods: {len(all_results)}\n\n")
        f.write("method\tPSNR\tSSIM\tSAM\tLPIPS\tFID\n")
        for method_name, avg in all_results:
            f.write(f"{method_name}\t{avg['psnr']:.6f}\t{avg['ssim']:.6f}\t{avg['sam']:.6f}\t{avg['lpips']:.6f}\t{avg['fid']:.6f}\n")

    print(f"\n[All Done] summary -> {summary_all}")


# ===========================
#   Main (你只改这里)
# ===========================

if __name__ == "__main__":
    config = {
        "gt_dir": r"E:\beifen\Sen2_MTC实验结果\postprocess\GT",

        # ✅ 方式A：给 postprocess 根目录，自动识别其下每个算法子文件夹
        "pred_root": r"E:\beifen\Sen2_MTC实验结果\abalation_results",

        # （可选）只跑指定算法：例如只跑 WaveCloudNet 和 PMAA
        # "methods": ["WaveCloudNet", "PMAA"],

        # ✅ 输出根目录：每个算法一个子目录
        "output_root": r"./evaluation_results_multi_ablation",

        "lpips_net": "alex",
        "workers": 4,
        "use_cpu": False,

        # 云量分段统计（可选）
        "cloud_stats": True,
        "cloud_txt_path": r"D:\研究生\论文代码\paper3\region_cloudiness.txt",
        "cloud_bins": [0.0, 20.0, 40.0, 60.0, float("inf")],
    }

    evaluate_multi(
        gt_dir=config["gt_dir"],
        pred_root=config.get("pred_root", None),
        pred_dirs=config.get("pred_dirs", None),   # 方式B：也可以手动给列表
        output_root=config["output_root"],
        lpips_net=config["lpips_net"],
        workers=config["workers"],
        use_cpu=config["use_cpu"],
        cloud_stats=config["cloud_stats"],
        cloud_txt_path=config["cloud_txt_path"],
        cloud_bins=config["cloud_bins"],
        methods=config.get("methods", None),
    )
