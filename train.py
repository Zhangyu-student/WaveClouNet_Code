import torch.nn.functional as F
from loss import MultiLoss
from setup_utils import create_model, create_dataloaders
from tqdm import tqdm  # 导入进度条库
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from metrics import mae, calculate_sam_rgb, psnr_skimage, process_rgb, ssim_skimage
from training_utils import *
import csv
import os
import random, numpy as np


def validate_model(model, test_loader, loss_fn, config, device, loss_keys, epoch, writer=None):
    """执行模型验证"""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_val_loss = 0.0
    total_psnr = 0.0
    total_sam = 0.0
    total_ssim = 0.0
    val_loss_components = {k: 0.0 for k in loss_keys}
    test_count = 0

    valid_pbar = tqdm(enumerate(test_loader),
                      total=len(test_loader),
                      desc=f'验证 Epoch {epoch}',
                      position=1,
                      leave=True,
                      colour='yellow')

    with torch.no_grad():
        total_samples = 0  # 初始化
        for batch_idx, ret in valid_pbar:
            cloudy_seq = ret["cond_image"].to(device)
            clean_target = ret["gt_image"].to(device)
            output, _, _ = model(cloudy_seq)

            # 计算验证损失
            batch_loss, batch_loss_dict = loss_fn(output, clean_target)
            total_val_loss += batch_loss.item()

            # 累加各项验证损失
            for k in loss_keys:
                val_loss_components[k] += batch_loss_dict[k]

            # 计算基本指标
            batch_mse = F.mse_loss(output, clean_target)
            batch_mae = mae(output, clean_target)
            total_mse += batch_mse.item()
            total_mae += batch_mae
            test_count += 1

            # 计算PSNR和SAM
            batch_size = output.size(0)
            total_samples += batch_size  # 累加实际样本数
            for i in range(batch_size):
                pred_uint8 = process_rgb(output[i].cpu(), config["dataset_type"])
                gt_uint8 = process_rgb(clean_target[i].cpu(), config["dataset_type"])
                total_psnr += psnr_skimage(gt_uint8, pred_uint8)
                total_sam += calculate_sam_rgb(gt_uint8, pred_uint8)
                total_ssim += ssim_skimage(gt_uint8, pred_uint8)  # 新增：计算SSIM
            # 更新验证进度条
            valid_pbar.set_postfix({
                'val_loss': f"{batch_loss.item():.4f}",
                'psnr': f"{total_psnr / total_samples:.2f}",
                'sam': f"{total_sam / total_samples:.2f}",
                'ssim': f"{total_ssim / total_samples:.4f}"
            })

    # 计算平均值
    avg_val_loss = total_val_loss / test_count
    avg_val_components = {k: v / test_count for k, v in val_loss_components.items()}
    avg_mse = total_mse / test_count
    avg_mae = total_mae / test_count
    avg_psnr = total_psnr / total_samples
    avg_sam = total_sam / total_samples
    avg_ssim = total_ssim / total_samples

    # 打印详细的验证结果
    print("\n" + "=" * 60)
    print(f"验证结果 Epoch {epoch}:")
    print(f"验证损失: {avg_val_loss:.4f} | MSE: {avg_mse:.4f} | MAE: {avg_mae:.4f}")
    print(f"PSNR: {avg_psnr:.2f} dB | SAM: {avg_sam:.2f}° | SSIM: {avg_ssim:.4f}")
    print("损失分量:")
    for k, v in avg_val_components.items():
        print(f"  {k}: {v:.6f}")
    print("=" * 60 + "\n")

    # 返回结果
    results = {
        'val_loss': avg_val_loss,
        'val_components': avg_val_components,
        'val_mse': avg_mse,
        'val_mae': avg_mae,
        'val_psnr': avg_psnr,
        'val_sam': avg_sam,
        'val_ssim': avg_ssim
    }

    return results


def log_results(csv_writer, epoch, train_loss, results, train_components):
    """记录结果到CSV并实时打印"""
    # 准备日志行
    log_row = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': results['val_loss'],
        'val_mse': results['val_mse'],
        'val_mae': results['val_mae'],
        'val_psnr': results['val_psnr'],
        'val_sam': results['val_sam'],
        'val_ssim': results['val_ssim']  # 新增
    }

    # 添加训练损失分量
    for k, v in train_components.items():
        log_row[k] = v

    # 添加验证损失分量
    for k, v in results['val_components'].items():
        log_row[f'val_{k}'] = v

    # 写入CSV并立即刷新
    csv_writer.writerow(log_row)

    # 实时打印日志
    print(f"Epoch {epoch} 日志已保存: "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {results['val_loss']:.4f}, "
          f"Val PSNR: {results['val_psnr']:.2f} dB")


def main():
    # ============ 固定随机种子，保证可复现性 ============
    seed = 2025  # 可自行设定，比如 42 / 2025 / 3407 等
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 训练配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_root': r'M:\多时相云修复数据集\s2_asiaWest\S2PatchDataset',
        'batch_size': 1,
        'num_epochs': 300,
        'valid_interval': 1,
        'save_dir': './checkpoints',
        'vis_dir': './visualizations',
        'log_file': './train_logs_landsat.csv',
        'log_dir': './runs',
        'pretrained_path': './checkpoints/model_epoch_10000.pth',
        'dataset_type': 's2asiawest'
    }

    # config = {
    #     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    #     'data_root': 'F:\SENMS_NEW\landsat_dataset',
    #     'batch_size': 1,
    #     'num_epochs': 300,
    #     'valid_interval': 1,
    #     'save_dir': './checkpoints',
    #     'vis_dir': './visualizations',
    #     'log_file': './train_logs_landsat.csv',
    #     'log_dir': './runs',
    #     'pretrained_path': './checkpoints/model_epoch_10000.pth',
    #     'dataset_type': 'landsat'
    # }

    # 训练配置
    # config = {
    #     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    #     'data_root': '/root/autodl-tmp/data/dataset',
    #     'batch_size': 8,
    #     'num_epochs': 300,
    #     'valid_interval': 1,
    #     'save_dir': './checkpoints',
    #     'vis_dir': './visualizations',
    #     'log_file': './train_logs_new_no_fcac.csv',
    #     'log_dir': './runs',
    #     'pretrained_path': './checkpoints/model_epoch_10000.pth',
    #     'dataset_type': 'new_multi'
    # }

    # config = {
    #     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    #     'data_root': '/root/autodl-tmp/data/multipleImage',
    #     'batch_size': 8,
    #     'num_epochs': 300,
    #     'valid_interval': 1,
    #     'save_dir': './checkpoints',
    #     'vis_dir': './visualizations',
    #     'log_file': './train_logs_old.csv',
    #     'log_dir': './runs',
    #     'pretrained_path': './checkpoints/model_epoch_300.pth',
    #     'dataset_type': 'old_multi'
    # }

    # 创建输出目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['vis_dir'], exist_ok=True)

    # 设置TensorBoard
    writer = setup_tensorboard(config)

    # 设置CSV日志（实时写入模式）
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_mse', 'val_mae', 'val_psnr', 'val_sam', 'val_ssim']
    loss_keys = ['l1_loss', 'spec_loss', 'grad_loss', 'ssim_loss']
    # loss_keys = ['l1_loss', 'spec_loss', 'ssim_loss']

    # 添加损失分量到字段名
    for k in loss_keys:
        fieldnames.append(k)
        fieldnames.append(f'val_{k}')

    with open(config['log_file'], 'w', newline='') as csv_file:  # 使用'w'模式
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()  # 确保写入表头
        csv_file.flush()

        # 1. 创建数据加载器
        train_loader, test_loader = create_dataloaders(config)

        # 2. 创建模型
        model = create_model(config)

        # 创建优化器和调度器
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)

        device = torch.device(config['device'])
        # New_dataset
        loss_fn = MultiLoss(device, alpha=5, gamma=1, delta=1)

        # Old_dataset
        # loss_fn = MultiLoss(device, alpha=1, gamma=1, delta=1)
        print("损失函数实例创建成功")

        # 数值初始化
        best_mse = float('inf')
        global_step = 0

        # 训练循环
        # 创建总的训练进度条
        epoch_pbar = tqdm(range(1, config['num_epochs'] + 1),
                          desc="总训练进度",
                          position=0,
                          leave=True,
                          colour='green')

        for epoch in epoch_pbar:
            model.train()
            total_loss = 0.0
            # 初始化损失项累加器（使用固定名称）
            loss_components = {k: 0.0 for k in loss_keys}

            # 创建批次级进度条
            batch_pbar = tqdm(enumerate(train_loader),
                              total=len(train_loader),
                              desc=f'训练 Epoch {epoch}/{config["num_epochs"]}',
                              position=1,
                              leave=False,
                              colour='blue')

            # 训练阶段
            for batch_idx, ret in batch_pbar:
                cloudy_seq = ret["cond_image"].to(config['device'])
                clean_target = ret["gt_image"].to(config['device'])

                # 前向传播
                output, _, _ = model(cloudy_seq)

                # 计算损失
                loss, loss_dict = loss_fn(output, clean_target)

                # 累加各项损失
                total_loss += loss.item()
                for k in loss_keys:
                    loss_components[k] += loss_dict[k]

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # TensorBoard记录
                if batch_idx % 10 == 0:
                    writer.add_scalar('Loss/total', loss.item(), global_step)
                    for name, value in loss_dict.items():
                        writer.add_scalar(f'Loss/train_{name}', value, global_step)

                global_step += 1

                # 在进度条中更新损失信息
                loss_str = " | ".join([f"{k[:5]}:{loss_dict[k]:.4f}" for k in loss_keys])
                batch_pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    'loss_comp': loss_str
                })

            # 关闭批次级进度条
            batch_pbar.close()

            # 计算平均训练损失
            avg_train_loss = total_loss / len(train_loader)
            avg_loss_components = {k: v / len(train_loader) for k, v in loss_components.items()}

            # 更新总进度条信息
            epoch_pbar.set_postfix({
                'train_loss': f"{avg_train_loss:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

            print(f'\nEpoch [{epoch}/{config["num_epochs"]}] 平均训练损失: {avg_train_loss:.4f}')
            print('-' * 40)
            print("训练损失分量:")
            for k, v in avg_loss_components.items():
                print(f"  {k:<12}: {v:.6f}")
                writer.add_scalar(f'Loss/train_epoch_{k}', v, epoch)
            print('-' * 40)

            writer.add_scalar('Loss/train_total_epoch', avg_train_loss, epoch)

            # 更新学习率
            scheduler.step()
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

            # 验证阶段
            if epoch % config['valid_interval'] == 0 or epoch == config['num_epochs']:
                results = validate_model(
                    model, test_loader, loss_fn, config,
                    device, loss_keys, epoch, writer
                )
                # 保存模型和记录结果
                new_best_mse = save_model(model, epoch, config, results, best_mse)
                best_mse = new_best_mse  # 更新本地的 best_mse

                log_results(writer_csv, epoch, avg_train_loss, results, avg_loss_components)
                csv_file.flush()  # 确保日志立即写入磁盘

                # 可视化
                visualize_results(model, test_loader, config, epoch, writer)

        # 清理资源
        csv_file.close()
        writer.close()
        print("🎉🎉🎉🎉 训练完成!")


if __name__ == '__main__':
    main()