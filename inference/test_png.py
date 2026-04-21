import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import Sen2_MTC_New_Multi
from models.WaveCloudNet import WaveDH
from lpips import LPIPS
from cleanfid import fid
from visualize import visualize_comparison

"""
此代码旨在复现其他论文针对Sen2_Multi的指标对比方式
"""

def process_rgb(image_tensor):
    """
    将模型输出张量转换为RGB图像的处理流程

    参数:
        image_tensor (torch.Tensor): 模型输出张量，形状为[3, H, W]

    返回:
        np.ndarray: uint8类型的RGB图像，形状为[H, W, 3]
    """
    # 反归一化到0-10000范围
    image = image_tensor * 0.5 + 0.5
    image = image * 10000
    image = torch.clamp(image, 0, 10000)

    # 转换为numpy数组并调整维度
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # 提取RGB通道并截断
    r = np.clip(image_np[:, :, 0], 0, 2000)
    g = np.clip(image_np[:, :, 1], 0, 2000)
    b = np.clip(image_np[:, :, 2], 0, 2000)

    # 组合RGB
    rgb = np.dstack((r, g, b))

    # 归一化到0-255范围
    rgb_min = np.min(rgb)
    rgb = rgb - rgb_min
    rgb_max = np.max(rgb)

    if rgb_max == 0:
        rgb = 255 * np.ones_like(rgb)
    else:
        rgb = 255 * (rgb / rgb_max)

    # 处理可能的NaN值
    rgb = np.nan_to_num(rgb, nan=np.nanmean(rgb))
    return rgb.astype(np.uint8), rgb.astype(np.float32)  # 返回uint8和float32两个版本


# 计算SAM（光谱角映射器）基于RGB波段
def calculate_sam_rgb(gt_rgb, pred_rgb):
    """
    计算预测RGB图像和真实RGB图像之间的光谱角映射器（SAM）值

    参数:
        gt_rgb (np.ndarray): 真实RGB图像, 形状为[H, W, 3]
        pred_rgb (np.ndarray): 预测RGB图像, 形状为[H, W, 3]

    返回:
        float: 以度为单位的SAM值
    """
    # 确保形状正确
    if gt_rgb.shape[-1] != 3 or pred_rgb.shape[-1] != 3:
        raise ValueError("Input images must have 3 channels (RGB)")

    # 确保数据类型是浮点型
    gt_rgb = gt_rgb.astype(np.float32)
    pred_rgb = pred_rgb.astype(np.float32)

    # 展平为像素光谱向量
    gt_flat = gt_rgb.reshape(-1, 3)
    pred_flat = pred_rgb.reshape(-1, 3)

    # 计算点积
    dot_product = np.sum(gt_flat * pred_flat, axis=1)

    # 计算范数
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)

    # 避免除以零
    valid_mask = (norm_gt > 1e-6) & (norm_pred > 1e-6)
    dot_product = dot_product[valid_mask]
    norm_gt = norm_gt[valid_mask]
    norm_pred = norm_pred[valid_mask]

    # 计算余弦相似度
    cos_theta = dot_product / (norm_gt * norm_pred)

    # 限制在[-1,1]范围内以防止数值误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算角度（弧度）
    angle_rad = np.arccos(cos_theta)

    # 转换角度为度
    angle_deg = np.degrees(angle_rad)

    # 计算有效像素的平均值
    return np.mean(angle_deg)


# 计算LPIPS（基于感知相似性的指标）
def calculate_lpips(lpips_model, gt_img, pred_img, device):
    """
    计算LPIPS指标（感知相似性）

    参数:
        lpips_model: 预训练的LPIPS模型
        gt_img: 真实图像numpy数组 (0-255范围)
        pred_img: 预测图像numpy数组 (0-255范围)
        device: 计算设备

    返回:
        float: LPIPS值
    """

    # 转换numpy数组为PyTorch张量
    def numpy_to_torch(img):
        # 归一化到[-1, 1]范围
        img_tensor = torch.from_numpy(img.astype(np.float32)) / 127.5 - 1.0
        # 调整为[C, H, W]格式
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        return img_tensor

    gt_tensor = numpy_to_torch(gt_img)
    pred_tensor = numpy_to_torch(pred_img)

    # 计算LPIPS
    with torch.no_grad():
        lpips_val = lpips_model(gt_tensor, pred_tensor).item()

    return lpips_val


# 保存图像用于FID计算
def save_images_for_fid(images, base_filename, fid_dir):
    """
    为FID计算保存图像

    参数:
        images: 图像列表 [H, W, 3]
        base_filename: 基础文件名
        fid_dir: FID图像保存目录
    """
    os.makedirs(fid_dir, exist_ok=True)

    # 为每个文件创建唯一标识符
    for i, img in enumerate(images):
        save_path = os.path.join(fid_dir, f"{base_filename}.png")
        Image.fromarray(img).save(save_path)


def inference(model_path, data_root, save_dir, batch_size=1, device='cuda', visualize_attention=False):
    """
    模型推理函数，保存去云结果图像为PNG文件，并计算基于RGB的PSNR、SSIM、SAM、LPIPS和FID指标

    参数:
        model_path (str): 模型权重文件路径
        data_root (str): 测试数据集根目录
        save_dir (str): 结果保存目录
        batch_size (int): 批处理大小
        device (str): 计算设备 ('cuda' 或 'cpu')
        visualize_attention (bool): 是否可视化注意力图
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建可视化目录
    if visualize_attention:
        vis_dir = os.path.join(save_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)

    # 初始化模型
    model = WaveDH(input_nc=3, output_nc=3, ngf=32, n_lo_b=2, n_bottles=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 初始化LPIPS模型
    lpips_model = LPIPS(net='alex').to(device)  # 使用AlexNet作为特征提取器
    lpips_model.eval()

    # 创建FID图像保存目录
    fid_gen_dir = os.path.join(save_dir, "fid_generated")
    fid_real_dir = os.path.join(save_dir, "fid_real")
    os.makedirs(fid_gen_dir, exist_ok=True)
    os.makedirs(fid_real_dir, exist_ok=True)

    # 加载测试数据集
    test_dataset = Sen2_MTC_New_Multi(data_root=data_root, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化指标累计器
    psnr_values = []
    ssim_values = []
    sam_values = []
    lpips_values = []
    all_filenames = []  # 存储所有文件名

    # 创建结果目录
    results_dir = os.path.join(save_dir, "metrics")
    os.makedirs(results_dir, exist_ok=True)

    # 创建详细结果CSV文件
    csv_path = os.path.join(results_dir, "detailed_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("filename,psnr,ssim,sam,lpips\n")

    # 批量推理
    with torch.no_grad():
        for batch_idx, ret in enumerate(test_loader):
            cloudy_seq = ret["cond_image"].to(device)
            gt_images = ret["gt_image"].to(device)
            filenames = ret["path"]  # 获取批次中的文件名

            # 存储所有文件名
            all_filenames.extend(filenames)

            # 修改模型调用以获取注意力图
            if visualize_attention:
                outputs, att_layer1, att_layer2 = model(cloudy_seq)
            else:
                outputs = model(cloudy_seq)[0]

            # 处理每个样本
            for sample_idx in range(outputs.shape[0]):
                # 提取预测和真实图像
                pred_tensor = outputs[sample_idx]
                gt_tensor = gt_images[sample_idx]
                cloudy_input = cloudy_seq[sample_idx]  # 获取当前样本的输入序列

                # 处理预测图像为RGB（获取uint8和float32两个版本）
                pred_uint8, pred_float = process_rgb(pred_tensor)
                gt_uint8, gt_float = process_rgb(gt_tensor)

                # 获取文件名
                base_filename = os.path.splitext(os.path.basename(filenames[sample_idx]))[0]
                save_path = os.path.join(save_dir, f"{base_filename}.png")
                save_path_gt = os.path.join(save_dir, f"{base_filename}_gt.png")

                # 保存PNG
                Image.fromarray(pred_uint8).save(save_path)
                Image.fromarray(gt_uint8).save(save_path_gt)

                # 保存图像用于FID计算
                save_images_for_fid([pred_uint8], base_filename, fid_gen_dir)
                save_images_for_fid([gt_uint8], base_filename, fid_real_dir)

                # 计算RGB指标
                psnr_val = psnr(gt_uint8, pred_uint8, data_range=255)
                ssim_val = ssim(gt_uint8, pred_uint8, data_range=255, channel_axis=-1)
                sam_val = calculate_sam_rgb(gt_uint8, pred_uint8)

                # 计算LPIPS指标
                try:
                    lpips_val = calculate_lpips(lpips_model, gt_uint8, pred_uint8, device)
                except Exception as e:
                    print(f"Error calculating LPIPS for {base_filename}: {str(e)}")
                    lpips_val = float('nan')

                # 记录指标
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                sam_values.append(sam_val)
                lpips_values.append(lpips_val)

                # 写入CSV文件
                with open(csv_path, "a") as f:
                    f.write(f"{base_filename},{psnr_val:.4f},{ssim_val:.4f},{sam_val:.4f},{lpips_val:.4f}\n")

                # 打印当前样本指标
                print(f"Image: {base_filename}")
                print(f"PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.4f} | SAM: {sam_val:.4f}° | LPIPS: {lpips_val:.4f}")

                # 调用可视化函数（如果需要）
                if visualize_attention:
                    # 调用可视化函数
                    visualize_comparison(
                        cloudy_input=cloudy_input,  # 多时相输入 [T, C, H, W]
                        output=pred_tensor,  # 模型输出 [C, H, W]
                        target=gt_tensor,  # 真实目标 [C, H, W]
                        epoch=base_filename,  # 使用批次索引作为epoch占位符
                        save_dir=vis_dir,  # 可视化保存目录
                        att_layer1=[a[sample_idx].cpu() for a in att_layer1],  # 第一层注意力图
                        att_layer2=[a[sample_idx].cpu() for a in att_layer2]  # 第二层注意力图
                    )
                    print(f"Visualization saved to {vis_dir}/epoch_{batch_idx:03d}.png")

                print("-----------------------------")

    # ... (其余代码保持不变) ...
    # 计算平均指标
    avg_psnr = np.nanmean(psnr_values)
    avg_ssim = np.nanmean(ssim_values)
    avg_sam = np.nanmean(sam_values)
    avg_lpips = np.nanmean(lpips_values)

    # 计算FID指标（需要安装clean-fid库）
    try:
        fid_value = fid.compute_fid(fid_real_dir, fid_gen_dir, device=device, num_workers=0)
    except Exception as e:
        print(f"Error calculating FID: {str(e)}")
        fid_value = float('nan')

    # 保存指标到文本文件
    metrics_path = os.path.join(results_dir, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test Results Summary\n")
        f.write(f"====================\n")
        f.write(f"Total Images: {len(psnr_values)}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average SAM: {avg_sam:.4f}°\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
        f.write(f"FID: {fid_value:.4f}\n\n")

    # 打印总结果
    print("\n=============================================")
    print("Test Metrics Summary:")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average SAM: {avg_sam:.4f}°")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"FID: {fid_value:.4f}")
    print("=============================================")
    print(f"PNG results saved to: {save_dir}")
    print(f"Metrics summary saved to: {metrics_path}")
    print(f"Detailed CSV saved to: {csv_path}")
    print(f"FID real images saved to: {fid_real_dir}")
    print(f"FID generated images saved to: {fid_gen_dir}")


if __name__ == "__main__":
    # 配置参数
    config = {
        "model_path": r"D:\研究生\论文代码\paper3\checkpoints\model_ablation\Sen_New\train_logs_loss_new_no_cnlm.pth",
        "data_root": r"E:\beifen\SENMS_NEW\CTGAN\CTGAN\Sen2_MTC\dataset",
        "save_dir": r"D:\研究生\论文代码\paper3\inference_results\new\model_ablation\inference_results_new_no_cnlm",
        "batch_size": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "visualize_attention": True  # 添加可视化标志位
    }
    # 执行推理
    inference(
        model_path=config["model_path"],
        data_root=config["data_root"],
        save_dir=config["save_dir"],
        batch_size=config["batch_size"],
        device=config["device"],
        visualize_attention = config["visualize_attention"]  # 传递可视化标志
    )
