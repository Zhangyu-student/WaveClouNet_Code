import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    gauss = torch.tensor([torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
                          for x in range(window_size)], dtype=torch.float)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    # 检查图像数据范围，确保在[0,1]之间
    max_val = 1

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # 添加数值稳定性保护
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    C3 = 1e-8  # 额外的稳定性常数

    # 使用安全的除法操作
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + C3

    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class MultiLoss(nn.Module):
    def __init__(self, device, alpha=0.3, gamma=0.05, delta=0.05):
        super().__init__()
        self.alpha = alpha  # 光谱损失权重
        self.gamma = gamma  # 梯度损失权重
        self.delta = delta  # SSIM损失权重

        # 确保设备设置正确
        self.device = device

    def spectral_loss(self, output, target):
        """光谱一致性损失"""
        # 确保不会出现除以零的情况
        dot_product = torch.sum(output * target, dim=1)
        norm_output = torch.norm(output, dim=1, keepdim=False)
        norm_target = torch.norm(target, dim=1, keepdim=False)

        # 添加小量防止数值不稳定
        cos_theta = dot_product / (norm_output * norm_target + 1e-6)
        # 确保cos_theta在有效的[-1,1]范围内
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        return torch.mean(torch.acos(cos_theta))

    def gradient_loss(self, output, target):
        # 同理，梯度在 [0,1] 域内计算更匹配 SSIM 的尺度
        out01 = (output + 1.0) * 0.5
        tgt01 = (target + 1.0) * 0.5
        odx, ody = self._image_gradients(out01)
        tdx, tdy = self._image_gradients(tgt01)
        return F.l1_loss(odx, tdx) + F.l1_loss(ody, tdy)

    def _image_gradients(self, img):
        """计算图像的水平和垂直梯度"""
        # 添加填充以保持原始尺寸
        pad_x = (0, 0, 1, 0)  # (左, 右, 上, 下)
        pad_y = (0, 1, 0, 0)

        dx = F.pad(img, pad_y, mode='replicate')[:, :, :, 1:] - F.pad(img, pad_y, mode='replicate')[:, :, :, :-1]
        dy = F.pad(img, pad_x, mode='replicate')[:, :, 1:, :] - F.pad(img, pad_x, mode='replicate')[:, :, :-1, :]

        return dx, dy

    def ssim_loss(self, img1, img2, window_size=11, size_average=True):
        # 将 [-1,1] -> [0,1]
        img1_01 = (img1 + 1.0) * 0.5
        img2_01 = (img2 + 1.0) * 0.5
        ssim_value = ssim(img1_01, img2_01, window_size, size_average)  # C1/C2 对应 max_val=1
        return 1 - ssim_value

    def forward(self, output, target):
        # 基础L1损失
        l1_loss = F.l1_loss(output, target)

        # 光谱损失
        spec_loss = self.spectral_loss(output, target)

        # 梯度损失
        grad_loss = self.gradient_loss(output, target)

        # SSIM损失 (1-SSIM)
        ssim_val_loss = self.ssim_loss(output, target)

        # 组合所有损失项
        total_loss = (l1_loss +
                      self.alpha * spec_loss +
                      self.gamma * grad_loss +
                      self.delta * ssim_val_loss)

        # 返回损失值字典
        return total_loss, {
            "l1_loss": l1_loss.item(),
            "spec_loss": spec_loss.item(),
            "grad_loss": grad_loss.item(),
            "ssim_loss": ssim_val_loss.item()
        }


def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建损失函数实例
    loss_fn = MultiLoss(device, alpha=0.3, gamma=0.05, delta=0.05)
    print("损失函数实例创建成功")

    # 模拟输入数据
    batch_size = 2
    channels = 3
    height, width = 256, 256

    # 创建模拟输出 (模型预测)
    output = torch.rand(batch_size, channels, height, width).to(device)
    print(f"模拟输出形状: {output.shape}")

    # 创建模拟目标 (真实图像)
    # 添加一些相关性，使loss计算更有意义
    target = output.clone().detach()
    target += torch.randn_like(target) * 0.1  # 添加一些噪声
    target = torch.clamp(target, 0, 1)  # 确保值在[0,1]范围内
    print(f"模拟目标形状: {target.shape}")

    print("\n计算损失...")

    # 计算损失
    total_loss, loss_dict = loss_fn(output, target)

    # 打印结果
    print("\n损失计算结果:")
    print(f"总损失: {total_loss.item():.6f}")
    print(f"L1损失: {loss_dict['l1_loss']:.6f}")
    print(f"光谱损失: {loss_dict['spec_loss']:.6f}")
    print(f"梯度损失: {loss_dict['grad_loss']:.6f}")
    print(f"SSIM损失: {loss_dict['ssim_loss']:.6f}")

    # 测试特殊情况
    print("\n测试特殊情况 - 相同输入输出:")

    # 输出和目标相同
    same_loss, same_loss_dict = loss_fn(output, output.clone())

    print(f"总损失 (相同输入输出): {same_loss.item():.6f}")
    print(f"L1损失 (相同输入输出): {same_loss_dict['l1_loss']:.6f} (应为0)")
    print(f"光谱损失 (相同输入输出): {same_loss_dict['spec_loss']:.6f} (应为0)")
    print(f"梯度损失 (相同输入输出): {same_loss_dict['grad_loss']:.6f} (应为0)")
    print(f"SSIM损失 (相同输入输出): {same_loss_dict['ssim_loss']:.6f} (应为0)")

    # 测试完全不同的输入
    print("\n测试特殊情况 - 完全不同的输入输出:")

    random_target = torch.rand(batch_size, channels, height, width).to(device)
    random_loss, random_loss_dict = loss_fn(output, random_target)

    print(f"总损失 (完全不同的输入输出): {random_loss.item():.6f} (应较大)")
    print(f"L1损失 (完全不同的输入输出): {random_loss_dict['l1_loss']:.6f} (应约为0.5)")
    print(f"光谱损失 (完全不同的输入输出): {random_loss_dict['spec_loss']:.6f} (应较大)")
    print(f"梯度损失 (完全不同的输入输出): {random_loss_dict['grad_loss']:.6f} (应较大)")
    print(f"SSIM损失 (完全不同的输入输出): {random_loss_dict['ssim_loss']:.6f} (应接近1)")

    # 测试数据类型一致性
    print("\n测试数据类型一致性:")
    print(f"总损失类型: {type(total_loss)} (应为torch.Tensor)")
    print(f"损失字典值类型: {type(loss_dict['l1_loss'])} (应为float)")

    # 测试数值稳定性
    print("\n测试数值稳定性 - 零输入:")

    zero_output = torch.zeros(batch_size, channels, height, width).to(device)
    zero_target = torch.zeros_like(zero_output)
    zero_loss, zero_loss_dict = loss_fn(zero_output, zero_target)

    print(f"零输入的总损失: {zero_loss.item():.6f} (应为0)")

    # 测试梯度计算
    print("\n测试梯度计算:")

    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x):
            return torch.sigmoid(self.conv(x))

    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 前向传播
    pred = model(output)
    loss, _ = loss_fn(pred, target)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("梯度计算完成，无错误")


if __name__ == "__main__":
    main()