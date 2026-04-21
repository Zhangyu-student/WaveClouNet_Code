import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def process_rgb(image_tensor, dataset_name="new_multi"):
    """
    将模型输出张量转换为RGB图像的处理流程

    参数:
        image_tensor (torch.Tensor): 模型输出张量，形状为[3, H, W]

    返回:
        np.ndarray: uint8类型的RGB图像，形状为[H, W, 3]
    """
    if dataset_name == "new_multi":
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
        return rgb.astype(np.uint8)

    elif dataset_name == "old_multi":
        # 反归一化到0-1范围
        image = image_tensor * 0.5 + 0.5
        image = torch.clamp(image, 0, 1)

        # 转换为numpy数组并调整维度
        image_np = image.permute(1, 2, 0).cpu().numpy()

        # 直接归一化到0-255范围
        rgb = np.clip(image_np, 0, 1)  # 确保值在[0,1]范围内
        rgb = 255 * rgb  # 缩放至[0,255]

        # 处理可能的NaN值
        rgb = np.nan_to_num(rgb, nan=np.nanmean(rgb))
        return rgb.astype(np.uint8)

    elif dataset_name == "landsat":
        # Landsat (假设C=7): 提取 4, 3, 2 波段 (索引 3, 2, 1) 作为 RGB (R=4, G=3, B=2)

        # 1. 反归一化到0-1范围
        # 注意: 这里的反归一化操作应用于整个张量
        image = image_tensor * 0.5 + 0.5
        image = torch.clamp(image, 0, 1)

        # 2. 提取所需的波段 (4, 3, 2 对应索引 3, 2, 1)
        # 提取 3, 2, 1 索引的通道，并按 R, G, B 的顺序排列
        # image[3:4, ...] 是第4波段 (R), image[2:3, ...] 是第3波段 (G), image[1:2, ...] 是第2波段 (B)
        # 使用 torch.cat 将它们沿着通道维度 (dim=0) 拼接起来
        rgb_channels = torch.cat([
            image[3:4, ...],  # R: Band 4 (index 3)
            image[2:3, ...],  # G: Band 3 (index 2)
            image[1:2, ...]  # B: Band 2 (index 1)
        ], dim=0)  # 新张量形状为 [3, H, W]

        # 3. 转换为numpy数组并调整维度
        # permute(1, 2, 0) 将 [C, H, W] 转换为 [H, W, C]
        image_np = rgb_channels.permute(1, 2, 0).cpu().numpy()

        # 4. 直接归一化到0-255范围
        # 由于反归一化后值已经在 [0, 1] 范围内
        rgb = np.clip(image_np, 0, 1)  # 确保值在[0,1]范围内
        rgb = 255 * rgb  # 缩放至[0,255]

        # 5. 处理可能的NaN值并转换为uint8
        rgb = np.nan_to_num(rgb, nan=np.nanmean(rgb))
        return rgb.astype(np.uint8)

    elif dataset_name == "s2asiawest":
        # Landsat (假设C=13): 提取 4, 3, 2 波段 (索引 3, 2, 1) 作为 RGB (R=4, G=3, B=2)

        # 1. 反归一化到0-1范围
        # 注意: 这里的反归一化操作应用于整个张量
        image = image_tensor * 0.5 + 0.5
        image = torch.clamp(image, 0, 1)

        # 2. 提取所需的波段 (4, 3, 2 对应索引 3, 2, 1)
        # 提取 3, 2, 1 索引的通道，并按 R, G, B 的顺序排列
        # image[3:4, ...] 是第4波段 (R), image[2:3, ...] 是第3波段 (G), image[1:2, ...] 是第2波段 (B)
        # 使用 torch.cat 将它们沿着通道维度 (dim=0) 拼接起来
        rgb_channels = torch.cat([
            image[3:4, ...],  # R: Band 4 (index 3)
            image[2:3, ...],  # G: Band 3 (index 2)
            image[1:2, ...]  # B: Band 2 (index 1)
        ], dim=0)  # 新张量形状为 [3, H, W]

        # 3. 转换为numpy数组并调整维度
        # permute(1, 2, 0) 将 [C, H, W] 转换为 [H, W, C]
        image_np = rgb_channels.permute(1, 2, 0).cpu().numpy()

        # 4. 直接归一化到0-255范围
        # 由于反归一化后值已经在 [0, 1] 范围内
        rgb = np.clip(image_np, 0, 1)  # 确保值在[0,1]范围内
        rgb = 255 * rgb  # 缩放至[0,255]

        # 5. 处理可能的NaN值并转换为uint8
        rgb = np.nan_to_num(rgb, nan=np.nanmean(rgb))
        return rgb.astype(np.uint8)

    else:
        # 如果 dataset_name 不匹配任何已知项，可以抛出错误或返回默认值
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


# MAE计算函数
def mae(output, target):
    """计算平均绝对误差(MAE)"""
    return F.l1_loss(output, target).item()

def calculate_sam_rgb(gt_rgb, pred_rgb):
    """计算RGB图像的SAM（与测试脚本一致）"""
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


def psnr_skimage(gt, pred):
    """封装PSNR计算"""
    return peak_signal_noise_ratio(gt, pred, data_range=255)

def ssim_skimage(gt, pred):
    """封装SSIM计算"""
    return structural_similarity(gt, pred, data_range=255, channel_axis=-1)