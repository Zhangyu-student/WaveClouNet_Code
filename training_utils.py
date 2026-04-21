# training_utils.py
import os
import csv
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from visualize import visualize_comparison
from torch.utils.data import DataLoader

def setup_tensorboard(config):
    """设置TensorBoard日志记录器"""
    os.makedirs(config['log_dir'], exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(config['log_dir'], f'run_{timestamp}'))
    print(f"TensorBoard日志保存到: {os.path.join(config['log_dir'], f'run_{timestamp}')}")
    return writer

def setup_csv_logging(config):
    """设置CSV日志记录"""
    os.makedirs(os.path.dirname(config['log_file']) or '.', exist_ok=True)
    csv_file = open(config['log_file'], 'w', newline='')
    fieldnames = ['epoch', 'train_loss', 'val_loss',
                  'val_mse', 'val_mae', 'val_psnr', 'val_sam',
                  'l1_loss', 'spec_loss', 'grad_loss', 'ssim_loss']
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()
    return csv_file, writer_csv

def create_optimizer(model, config):
    """创建优化器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    return optimizer

def create_scheduler(optimizer, config):
    """创建学习率调度器"""
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    return scheduler


def save_model(model, epoch, config, results, best_mse):
    """保存模型检查点"""
    ckpt_path = os.path.join(config['save_dir'], f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), ckpt_path)

    # 更新最佳模型
    if results['val_mse'] < best_mse:
        best_mse = results['val_mse']
        best_path = os.path.join(config['save_dir'], 'best.pth')
        torch.save(model.state_dict(), best_path)
    return best_mse

def log_results(writer_csv, epoch, train_loss, results, loss_components):
    """记录结果到CSV"""
    log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': results['val_loss'],
        'val_mse': results['val_mse'],
        'val_mae': results['val_mae'],
        'val_psnr': results['val_psnr'],
        'val_sam': results['val_sam']
    }
    # 添加损失组件
    for k, v in loss_components.items():
        log_data[k] = v
    writer_csv.writerow(log_data)


def visualize_results(model, test_loader, config, epoch, writer):
    model.eval()

    # --- 1. 确定要可视化的通道数量 ---
    # 如果是 Landsat，输入是 7 通道，但我们只可视化前 3 通道 (RGB或自定义组合)
    if config['dataset_type'] == 'landsat':
        vis_channels = 7  # 输入通道数
        # 我们只选择前 3 个通道进行可视化。
        # 注意：这假设 Landsat 的前 3 个通道是适合直接可视化的 (如 B4, B3, B2 组合)
        channel_indices = [0, 1, 2]
    elif config['dataset_type'] == 's2asiawest':
        vis_channels = 13  # 输入通道数
        # 我们只选择前 3 个通道进行可视化。
        # 注意：这假设 Landsat 的前 3 个通道是适合直接可视化的 (如 B4, B3, B2 组合)
        channel_indices = [1, 2, 3]
    else:
        # 对于 Sen2_MTC_New_Multi 和 MultipleDataset，输入就是 3 或 4 个通道，但输出是 3个通道的可视化
        vis_channels = 3  # 假设所有非Landsat数据集的GT是3通道RGB
        channel_indices = [0, 1, 2]

    with torch.no_grad():
        # 创建临时加载器实现随机采样
        temp_loader = DataLoader(
            test_loader.dataset,  # 使用相同数据集
            batch_size=config['batch_size'],  # 保持原批次大小
            shuffle=True,  # 关键：启用随机打乱
            num_workers=0,  # 避免多进程问题
            collate_fn=test_loader.collate_fn if hasattr(test_loader,
                                                         'collate_fn') else torch.utils.data.dataloader.default_collate
            # 保持数据整理方式
        )

        try:
            sample_batch = next(iter(temp_loader))
        except StopIteration:
            return  # 空数据集时安全退出

        test_input = sample_batch["cond_image"].to(config['device'])
        test_target = sample_batch["gt_image"].to(config['device'])

        # 确保输入cond_image的通道数与模型输入匹配，这里假设模型输入是 (B, T, C, H, W)
        # T=3 (时相数), C=通道数 (3或7)
        # model(test_input) 假设能正确处理 (B, 3, C, H, W)
        test_output, att_layer1, att_layer2 = model(test_input)

        # 随机选择批次中的一个样本
        random_idx = torch.randint(0, test_input.size(0), (1,)).item()

        # --- 2. 核心修改：通道选择和时相选择 ---

        # 1. 目标 (GT) 图像：只取前 K 个通道进行可视化
        # 注意：GT图像形状是 (C, H, W)
        target_vis = test_target[random_idx].cpu()[channel_indices, :, :]

        # 2. 输出 (Output) 图像：只取前 K 个通道进行可视化
        # 注意：Output图像形状是 (C, H, W)
        output_vis = test_output[random_idx].cpu()[channel_indices, :, :]

        # 3. 输入 (Input) 图像：选择第一个时相 (T1) 的前 K 个通道进行可视化
        # 注意：Input图像形状是 (T, C, H, W)
        # 这里选择第一个时相 [0] 进行可视化，并只取前 K 个通道
        input_vis = test_input[random_idx].cpu()[:, channel_indices, :, :]

        # ----------------------------------------

        vis_path = visualize_comparison(
            cloudy_input=input_vis,
            output=output_vis,
            target=target_vis,
            epoch=epoch,
            save_dir=config['vis_dir'],
            att_layer1=[a[random_idx].cpu() for a in att_layer1],
            att_layer2=[a[random_idx].cpu() for a in att_layer2]
        )

        if vis_path and os.path.exists(vis_path):
            try:
                from PIL import Image
                import numpy as np
                img = Image.open(vis_path)
                img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)
                writer.add_image(f'Comparison/epoch_{epoch}', img_tensor, epoch, dataformats='CHW')
            except Exception as e:
                print(f"无法将图像添加到TensorBoard: {e}")