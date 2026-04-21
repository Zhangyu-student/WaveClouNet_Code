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


# ===========================
#   Cloud map helpers
# ===========================

# 从文件名中提取区域ID：例如 01WFN_60009000（两位数字+3字母+_+数字）
REGION_RE = re.compile(r"(\d{2}[A-Z]{3}_[0-9]+)", re.IGNORECASE)

def extract_region_id(name: str):
    """
    从 filename(不带后缀) 中提取区域ID，用于和云量txt对齐
    """
    m = REGION_RE.search(name)
    return m.group(1).upper() if m else None

def load_cloud_map(txt_path: str):
    """
    读取云量txt：每行 'region\\tcloudiness' (tab/空格都可)
    返回 dict: region -> float
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
            region = parts[0].strip().upper()
            try:
                val = float(parts[1])
            except:
                continue
            cloud[region] = val
    return cloud

def assign_cloud_bin(val: float, bins, labels):
    """
    bins: e.g. [0,0.33,0.66,1.0]
    labels: e.g. ["low","mid","high"]
    """
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i+1]
        if (val >= left) and (val < right or (i == len(bins) - 2 and val <= right)):
            return labels[i]
    return None


# ===========================
#   Metrics helpers
# ===========================

def calculate_sam_rgb(gt_rgb, pred_rgb):
    """计算预测RGB图像和真实RGB图像之间的光谱角映射器（SAM）值"""
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
    """计算LPIPS指标（感知相似性）"""

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
    """为FID计算保存图像"""
    os.makedirs(save_dir, exist_ok=True)
    for img, filename in zip(images, filenames):
        base_name = os.path.splitext(filename)[0]
        Image.fromarray(img).save(os.path.join(save_dir, f"{base_name}.png"))


# ===========================
#   Dataset
# ===========================

class ImagePairDataset(Dataset):
    """用于成对图像评估的数据集（宽松匹配版）"""

    def __init__(self, gt_dir, pred_dir):
        self.gt_images = sorted(
            glob.glob(os.path.join(gt_dir, "*.png")) +
            glob.glob(os.path.join(gt_dir, "*.jpg")) +
            glob.glob(os.path.join(gt_dir, "*.jpeg"))
        )
        self.pred_images = sorted(
            glob.glob(os.path.join(pred_dir, "*.png")) +
            glob.glob(os.path.join(pred_dir, "*.jpg")) +
            glob.glob(os.path.join(pred_dir, "*.jpeg"))
        )
        self.gt_paths = []
        self.pred_paths = []
        self.filenames = []

        for pred_path in self.pred_images:
            pred_name = os.path.splitext(os.path.basename(pred_path))[0]
            matched_gt = next(
                (gt for gt in self.gt_images if pred_name in os.path.basename(gt)),
                None
            )
            if matched_gt:
                self.gt_paths.append(matched_gt)
                self.pred_paths.append(pred_path)
                self.filenames.append(pred_name)

        print(f"找到 {len(self.filenames)} 对匹配图像")

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

        return {
            "gt": gt_img,
            "pred": pred_img,
            "filename": self.filenames[idx]
        }


# ===========================
#   Evaluate
# ===========================

def evaluate(
    gt_dir, pred_dir, output_dir="evaluation_results",
    lpips_net="alex", workers=4, use_cpu=False,
    cloud_stats=False, cloud_txt_path=None, cloud_bins=None
):
    """执行图像质量评估 + （可选）按云量分段统计"""

    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    print(f"使用设备: {device}")

    os.makedirs(output_dir, exist_ok=True)

    config = {
        "gt_dir": gt_dir,
        "pred_dir": pred_dir,
        "output_dir": output_dir,
        "lpips_net": lpips_net,
        "workers": workers,
        "use_cpu": use_cpu,
        "cloud_stats": cloud_stats,
        "cloud_txt_path": cloud_txt_path,
        "cloud_bins": cloud_bins,
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"初始化LPIPS模型 (使用 {lpips_net} 网络)...")
    lpips_model = LPIPS(net=lpips_net).to(device)
    lpips_model.eval()

    print("准备数据集...")
    dataset = ImagePairDataset(gt_dir, pred_dir)
    if len(dataset) == 0:
        print("错误: 在指定目录中没有找到匹配的图像对!")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)

    metrics = {
        "psnr": [],
        "ssim": [],
        "sam": [],
        "lpips": [],
        "filenames": dataset.filenames
    }

    # ====== FID temp dirs ======
    fid_real_dir = os.path.join(output_dir, "fid_real")
    fid_pred_dir = os.path.join(output_dir, "fid_pred")

    for d in [fid_real_dir, fid_pred_dir]:
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)
        os.makedirs(d)

    fid_real_imgs, fid_pred_imgs, fid_filenames = [], [], []

    # ====== 云量分桶（可选）======
    cloud_map = None
    bins = cloud_bins if cloud_bins is not None else [0.0, 0.33, 0.66, 1.0]
    labels = ["low", "mid", "high"] if bins == [0.0, 0.33, 0.66, 1.0] else [f"bin{i}" for i in range(len(bins)-1)]
    bin_metrics = {lb: {"psnr": [], "ssim": [], "sam": [], "lpips": [], "n": 0} for lb in labels}
    missing_cloud = []
    # ===============================

    if cloud_stats:
        if (cloud_txt_path is None) or (not os.path.exists(cloud_txt_path)):
            print("错误: cloud_stats=True 但 cloud_txt_path 无效/不存在！")
            return
        cloud_map = load_cloud_map(cloud_txt_path)
        print(f"已加载云量映射: {len(cloud_map)} regions from {cloud_txt_path}")
        print(f"云量分段 bins={bins} labels={labels}")

    print("开始评估图像...")
    for batch in tqdm(dataloader, desc="评估图像"):
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
            ssim_val = ssim(
                gt_img, pred_img,
                channel_axis=-1,
                multichannel=True,
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

        if not np.isnan(psnr_val):
            print(f"\n图像: {filename}")
            print(f"PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.4f} | SAM: {sam_val:.4f}° | LPIPS: {lpips_val:.4f}")

        # ====== 云量分桶统计（可选）======
        if cloud_stats and (cloud_map is not None) and (not np.isnan(psnr_val)):
            region_id = extract_region_id(filename)
            if region_id is None or region_id not in cloud_map:
                missing_cloud.append(filename)
            else:
                cval = cloud_map[region_id]
                lb = assign_cloud_bin(cval, bins=bins, labels=labels)
                if lb is not None:
                    bin_metrics[lb]["psnr"].append(psnr_val)
                    bin_metrics[lb]["ssim"].append(ssim_val)
                    bin_metrics[lb]["sam"].append(sam_val)
                    bin_metrics[lb]["lpips"].append(lpips_val)
                    bin_metrics[lb]["n"] += 1
        # ===============================

    # ====== FID ======
    print("为FID计算准备图像...")
    save_images_for_fid(fid_real_imgs, fid_filenames, fid_real_dir)
    save_images_for_fid(fid_pred_imgs, fid_filenames, fid_pred_dir)

    print("计算FID指标...")
    try:
        fid_value = fid.compute_fid(fid_real_dir, fid_pred_dir, device=device, num_workers=0)
    except Exception as e:
        print(f"计算FID时出错: {str(e)}")
        fid_value = float('nan')

    avg_metrics = {
        "psnr": np.nanmean(metrics["psnr"]),
        "ssim": np.nanmean(metrics["ssim"]),
        "sam": np.nanmean(metrics["sam"]),
        "lpips": np.nanmean(metrics["lpips"]),
        "fid": fid_value
    }

    # ====== save detailed ======
    detail_path = os.path.join(output_dir, "detailed_metrics.csv")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write("filename,psnr,ssim,sam,lpips\n")
        for i, fn in enumerate(metrics["filenames"]):
            f.write(f"{fn},{metrics['psnr'][i]:.4f},{metrics['ssim'][i]:.4f},{metrics['sam'][i]:.4f},{metrics['lpips'][i]:.4f}\n")

    # ====== save summary ======
    summary_path = os.path.join(output_dir, "summary_metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("图像质量评估结果汇总\n")
        f.write("================================\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"真实图像目录: {gt_dir}\n")
        f.write(f"预测图像目录: {pred_dir}\n")
        f.write(f"总图像数: {len(dataset)}\n\n")
        f.write(f"平均 PSNR: {avg_metrics['psnr']:.4f} dB\n")
        f.write(f"平均 SSIM: {avg_metrics['ssim']:.4f}\n")
        f.write(f"平均 SAM: {avg_metrics['sam']:.4f}°\n")
        f.write(f"平均 LPIPS: {avg_metrics['lpips']:.4f}\n")
        f.write(f"FID: {avg_metrics['fid']:.4f}\n\n")

    # ====== save by-cloud summary (optional) ======
    if cloud_stats:
        bycloud_path = os.path.join(output_dir, "summary_metrics_by_cloud.txt")
        with open(bycloud_path, "w", encoding="utf-8") as f:
            f.write("按云量分段的图像质量评估汇总\n")
            f.write("================================\n")
            f.write(f"cloud_txt: {cloud_txt_path}\n")
            f.write(f"bins: {bins}\n\n")

            for lb in labels:
                n = bin_metrics[lb]["n"]
                if n == 0:
                    f.write(f"[{lb}] n=0\n\n")
                    continue
                f.write(f"[{lb}] n={n}\n")
                f.write(f"  PSNR : {float(np.mean(bin_metrics[lb]['psnr'])):.4f}\n")
                f.write(f"  SSIM : {float(np.mean(bin_metrics[lb]['ssim'])):.4f}\n")
                f.write(f"  SAM  : {float(np.mean(bin_metrics[lb]['sam'])):.4f}\n")
                f.write(f"  LPIPS: {float(np.mean(bin_metrics[lb]['lpips'])):.4f}\n\n")

            f.write(f"未能匹配云量的样本数: {len(missing_cloud)}\n")
            for x in missing_cloud[:50]:
                f.write(f"  {x}\n")

        print(f"分云量段结果已保存至: {bycloud_path}")

    # ====== print final ======
    print("\n" + "=" * 80)
    print("图像质量评估完成！结果汇总:")
    print("=" * 80)
    print(f"总图像数: {len(dataset)}")
    print(f"平均 PSNR: {avg_metrics['psnr']:.4f} dB")
    print(f"平均 SSIM: {avg_metrics['ssim']:.4f}")
    print(f"平均 SAM: {avg_metrics['sam']:.4f}°")
    print(f"平均 LPIPS: {avg_metrics['lpips']:.4f}")
    print(f"FID: {avg_metrics['fid']:.4f}")
    print("=" * 80)
    print(f"详细结果已保存至: {detail_path}")
    print(f"摘要结果已保存至: {summary_path}")
    print(f"配置信息已保存至: {os.path.join(output_dir, 'config.json')}")


# ===========================
#   Main
# ===========================

if __name__ == "__main__":
    config = {
        "gt_dir": r"E:\研究生\研究生科研学习\paper3\inference_results\fid_real",
        "pred_dir": r"E:\研究生\研究生科研学习\paper3\inference_results\fid_generated",
        "output_dir": "./evaluation_results",
        "lpips_net": "alex",
        "workers": 4,
        "use_cpu": False,

        # ====== 你要的标志位与云量txt ======
        "cloud_stats": True,  # True=按云量分段统计；False=只算整体平均
        "cloud_txt_path": r"E:\研究生\研究生科研学习\paper3\inference_results\visualization\att_layer1_raw\region_cloudiness.txt",
        "cloud_bins": [0.0, 0.30, 0.60, 1.0],  # 三档：low/mid/high（你可改）
    }

    evaluate(
        gt_dir=config["gt_dir"],
        pred_dir=config["pred_dir"],
        output_dir=config["output_dir"],
        lpips_net=config["lpips_net"],
        workers=config["workers"],
        use_cpu=config["use_cpu"],
        cloud_stats=config["cloud_stats"],
        cloud_txt_path=config["cloud_txt_path"],
        cloud_bins=config["cloud_bins"],
    )
