import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn.functional as F  # 添加这行导入

def visualize_comparison(cloudy_input, output, target, epoch, save_dir,
                         att_layer1=None, att_layer2=None):
    """扩展为三行五列可视化布局"""
    fig = plt.figure(figsize=(20, 12))

    def linear_stretch(img, percent=2):
        """
        对各通道进行2%线性拉伸
        参数：
            img: [C, H, W] 张量（取值范围任意）
            percent: 裁剪百分比
        返回：
            stretched_img: [H, W, C] numpy数组，取值范围[0,1]
        """
        # 转换为numpy数组并调整维度
        img_np = img.cpu().numpy()
        img_np = img_np.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]

        # 对每个通道单独处理
        stretched = np.zeros_like(img_np, dtype=np.float32)
        for c in range(img_np.shape[-1]):
            channel = img_np[..., c]

            # 计算裁剪阈值
            lower = np.percentile(channel, percent)
            upper = np.percentile(channel, 100 - percent)

            # 线性拉伸
            stretched_channel = np.clip((channel - lower) / (upper - lower), 0, 1)
            stretched[..., c] = stretched_channel

        return stretched

    # 第一行：输入图像
    for i in range(3):
        ax = fig.add_subplot(3, 5, i + 1)
        input_img = cloudy_input[i]
        ax.imshow(linear_stretch(input_img))
        ax.set_title(f'Input T{i + 1}')
        ax.axis('off')

    # 模型输出
    ax = fig.add_subplot(3, 5, 4)
    ax.imshow(linear_stretch(output))
    ax.set_title('Model Output')
    ax.axis('off')

    # 真实目标
    ax = fig.add_subplot(3, 5, 5)
    ax.imshow(linear_stretch(target))
    ax.set_title('Ground Truth')
    ax.axis('off')

    # 第二行：第一层注意力图
    if att_layer1 is not None:
        for i, att in enumerate(att_layer1):
            ax = fig.add_subplot(3, 5, 6 + i)

            # 确保注意力图是正确维度的张量
            if att.dim() == 3:  # [C, H, W]
                att = att[0]  # 取第一个通道 [H, W]

            # 上采样到原始尺寸 (创建4维张量: [batch=1, channels=1, H, W])
            att_4d = att.unsqueeze(0).unsqueeze(0) if att.dim() == 2 else att.unsqueeze(0)

            att_resized = F.interpolate(
                att_4d,
                size=cloudy_input.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # 转换为numpy数组
            att_np = att_resized[0, 0].cpu().numpy()
            ax.imshow(att_np, cmap='viridis')
            ax.set_title(f'T{i + 1} Layer1 Att')
            ax.axis('off')

    # 第三行：第二层注意力图
    if att_layer2 is not None:
        for i, att in enumerate(att_layer2):
            ax = fig.add_subplot(3, 5, 11 + i)

            # 确保注意力图是正确维度的张量
            if att.dim() == 3:  # [C, H, W]
                att = att[0]  # 取第一个通道 [H, W]

            # 上采样到原始尺寸
            att_4d = att.unsqueeze(0).unsqueeze(0) if att.dim() == 2 else att.unsqueeze(0)

            att_resized = F.interpolate(
                att_4d,
                size=cloudy_input.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # 转换为numpy数组
            att_np = att_resized[0, 0].cpu().numpy()
            ax.imshow(att_np, cmap='viridis')
            ax.set_title(f'T{i + 1} Layer2 Att')
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'), bbox_inches='tight', dpi=300)
    plt.close()