import os
import glob
import argparse
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


def get_filenames_only(file_paths):
    """从文件路径中提取不带后缀的文件名"""
    return [os.path.splitext(os.path.basename(f))[0] for f in file_paths]


def save_images_for_fid(images, filenames, save_dir):
    """为FID计算保存图像"""
    os.makedirs(save_dir, exist_ok=True)
    for img, filename in zip(images, filenames):
        base_name = os.path.splitext(filename)[0]
        Image.fromarray(img).save(os.path.join(save_dir, f"{base_name}.png"))


class ImagePairDataset(Dataset):
    """用于成对图像评估的数据集（宽松匹配版）"""

    def __init__(self, gt_dir, pred_dir):
        # 获取所有 PNG/JPG 图像文件
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
            # 寻找真值文件中包含预测名的项
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


def evaluate(gt_dir, pred_dir, output_dir="evaluation_results",
             lpips_net="alex", workers=4, use_cpu=False):
    """执行图像质量评估"""

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    print(f"使用设备: {device}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 记录配置
    config = {
        "gt_dir": gt_dir,
        "pred_dir": pred_dir,
        "output_dir": output_dir,
        "lpips_net": lpips_net,
        "workers": workers,
        "use_cpu": use_cpu,
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 初始化LPIPS模型
    print(f"初始化LPIPS模型 (使用 {lpips_net} 网络)...")
    lpips_model = LPIPS(net=lpips_net).to(device)
    lpips_model.eval()

    # 创建数据集
    print("准备数据集...")
    dataset = ImagePairDataset(gt_dir, pred_dir)

    # 如果没有匹配的文件，退出
    if len(dataset) == 0:
        print("错误: 在指定目录中没有找到匹配的图像对!")
        return

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)

    # 初始化指标存储
    metrics = {
        "psnr": [],
        "ssim": [],
        "sam": [],
        "lpips": [],
        "filenames": dataset.filenames
    }

    # 创建FID图像临时目录
    fid_real_dir = os.path.join(output_dir, "fid_real")
    fid_pred_dir = os.path.join(output_dir, "fid_pred")

    # 清空FID目录
    for d in [fid_real_dir, fid_pred_dir]:
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)
        os.makedirs(d)

    # 创建FID保存的列表
    fid_real_imgs = []
    fid_pred_imgs = []
    fid_filenames = []

    print("开始评估图像...")
    # 处理每对图像
    for batch in tqdm(dataloader, desc="评估图像"):
        gt_img = batch["gt"][0].numpy()
        pred_img = batch["pred"][0].numpy()
        filename = batch["filename"][0]

        # 添加FID图像
        fid_real_imgs.append(gt_img)
        fid_pred_imgs.append(pred_img)
        fid_filenames.append(filename)

        # 确保图像尺寸一致
        if gt_img.shape[:2] != pred_img.shape[:2]:
            print("尺寸不一致！")
            min_h = min(gt_img.shape[0], pred_img.shape[0])
            min_w = min(gt_img.shape[1], pred_img.shape[1])
            gt_img = gt_img[:min_h, :min_w]
            pred_img = pred_img[:min_h, :min_w]

        try:
            # 计算PSNR和SSIM
            psnr_val = psnr(gt_img, pred_img, data_range=255)
            ssim_val = ssim(gt_img, pred_img, data_range=255, channel_axis=-1)

            # 计算SAM
            sam_val = calculate_sam_rgb(gt_img, pred_img)

            # 计算LPIPS
            lpips_val = calculate_lpips(lpips_model, gt_img, pred_img, device)
        except Exception as e:
            print(f"计算 {filename} 的指标时出错: {str(e)}")
            # 使用默认值继续
            psnr_val = ssim_val = sam_val = lpips_val = float('nan')

        # 存储指标
        metrics["psnr"].append(psnr_val)
        metrics["ssim"].append(ssim_val)
        metrics["sam"].append(sam_val)
        metrics["lpips"].append(lpips_val)

        # 显示当前图像结果
        if not np.isnan(psnr_val):  # 仅在有有效值时打印
            print(f"\n图像: {filename}")
            print(f"PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.4f} | SAM: {sam_val:.4f}° | LPIPS: {lpips_val:.4f}")

    # 保存FID图像
    print("为FID计算准备图像...")
    save_images_for_fid(fid_real_imgs, fid_filenames, fid_real_dir)
    save_images_for_fid(fid_pred_imgs, fid_filenames, fid_pred_dir)

    # 计算FID指标
    print("计算FID指标...")
    try:
        fid_value = fid.compute_fid(fid_real_dir, fid_pred_dir, device=device, num_workers=0)
    except Exception as e:
        print(f"计算FID时出错: {str(e)}")
        fid_value = float('nan')

    # 计算平均指标
    avg_metrics = {
        "psnr": np.nanmean(metrics["psnr"]),
        "ssim": np.nanmean(metrics["ssim"]),
        "sam": np.nanmean(metrics["sam"]),
        "lpips": np.nanmean(metrics["lpips"]),
        "fid": fid_value
    }

    # 保存详细结果
    detail_path = os.path.join(output_dir, "detailed_metrics.csv")
    with open(detail_path, "w") as f:
        f.write("filename,psnr,ssim,sam,lpips\n")
        for i, filename in enumerate(metrics["filenames"]):
            f.write(
                f"{filename},{metrics['psnr'][i]:.4f},{metrics['ssim'][i]:.4f},{metrics['sam'][i]:.4f},{metrics['lpips'][i]:.4f}\n")

    # 保存摘要结果
    summary_path = os.path.join(output_dir, "summary_metrics.txt")
    with open(summary_path, "w") as f:
        f.write("图像质量评估结果汇总\n")
        f.write("================================\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"真实图像目录: {gt_dir}\n")
        f.write(f"预测图像目录: {pred_dir}\n")
        f.write(f"总图像数: {len(dataset)}\n")
        f.write("\n")
        f.write(f"平均 PSNR: {avg_metrics['psnr']:.4f} dB\n")
        f.write(f"平均 SSIM: {avg_metrics['ssim']:.4f}\n")
        f.write(f"平均 SAM: {avg_metrics['sam']:.4f}°\n")
        f.write(f"平均 LPIPS: {avg_metrics['lpips']:.4f}\n")
        f.write(f"FID: {avg_metrics['fid']:.4f}\n\n")

    # 打印最终结果
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


# 直接在脚本中设置路径 - 根据你的需要修改这些路径
if __name__ == "__main__":
    # 配置参数 - 在这里设置你的路径
    config = {
        # "gt_dir": r"D:\研究生\论文代码\paper3\inference_results\old\inference_results_old\fid_real",  # 替换为真实图像目录
        "gt_dir": r"E:\beifen\Sen2_MTC_old实验结果\postprocess\GT_jpg",
        # "pred_dir": r"D:\研究生\论文代码\paper3\inference_results\new\inference_results_all\fid_generated",  # 替换为预测图像目录
        "pred_dir": r"E:\beifen\Sen2_MTC_old实验结果\postprocess\UnCRTainTS",  # 替换为预测图像目录
        "output_dir": "./evaluation_results",  # 结果输出目录
        "lpips_net": "alex",  # LPIPS网络类型: alex/vgg/squeeze
        "workers": 4,  # 数据加载线程数
        "use_cpu": False  # 是否强制使用CPU
    }

    # 执行评估
    evaluate(
        gt_dir=config["gt_dir"],
        pred_dir=config["pred_dir"],
        output_dir=config["output_dir"],
        lpips_net=config["lpips_net"],
        workers=config["workers"],
        use_cpu=config["use_cpu"]
    )
