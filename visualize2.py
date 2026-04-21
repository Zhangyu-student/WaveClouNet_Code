import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

def visualize_comparison(
    cloudy_input, output, target, epoch, save_dir,
    att_layer1=None, att_layer2=None,
    percent_stretch=2,
    cmap="viridis",
    save_inputs_T=3,   # 你原来是3个时相；若想保存全部就设为None
):
    """
    单独保存(无标题)：
    - 输入时相图：inputs/epoch_xx_T1.png ...
    - 输出：output/epoch_xx_output.png
    - GT：gt/epoch_xx_gt.png
    - 注意力热力图（归一化到[0,1]）：att_layer1_raw/, att_layer2_raw/
    不做 overlay，不加 title。
    """

    os.makedirs(save_dir, exist_ok=True)

    # 子目录（更清晰）
    inputs_dir = os.path.join(save_dir, "inputs")
    output_dir = os.path.join(save_dir, "output")
    gt_dir     = os.path.join(save_dir, "gt")
    att1_dir   = os.path.join(save_dir, "att_layer1_raw")
    att2_dir   = os.path.join(save_dir, "att_layer2_raw")

    for d in [inputs_dir, output_dir, gt_dir, att1_dir, att2_dir]:
        os.makedirs(d, exist_ok=True)

    epoch_tag = str(epoch)

    # ---------- helpers ----------
    def linear_stretch(img_chw, percent=2):
        """对各通道进行percent%线性拉伸，输出[H,W,C]，范围[0,1]"""
        img_np = img_chw.detach().float().cpu().numpy()
        img_np = img_np.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
        stretched = np.zeros_like(img_np, dtype=np.float32)

        for c in range(img_np.shape[-1]):
            channel = img_np[..., c]
            lower = np.percentile(channel, percent)
            upper = np.percentile(channel, 100 - percent)
            denom = (upper - lower)
            if denom < 1e-8:
                stretched[..., c] = np.clip(channel, 0, 1)
            else:
                stretched[..., c] = np.clip((channel - lower) / denom, 0, 1)
        return stretched

    def norm01(x, eps=1e-8):
        """归一化到[0,1]"""
        x = x.astype(np.float32)
        mn, mx = float(np.min(x)), float(np.max(x))
        if (mx - mn) < eps:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn + eps)

    def att_to_hw(att_tensor, out_hw):
        """
        将注意力tensor转成 [H,W] numpy，并上采样到 out_hw
        支持：
        - [H,W]
        - [C,H,W] / [1,H,W]（取第0通道）
        - [1,1,H,W] / [B,1,H,W]（你传进来应该是单样本，所以这里也兼容）
        """
        att = att_tensor.detach()
        if att.dim() == 4:
            # [B, C, h, w] 或 [1,1,h,w]
            att = att[0, 0]
        elif att.dim() == 3:
            # [C,h,w]
            att = att[0]
        elif att.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported attention dim: {att.dim()}")

        att_4d = att.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        att_resized = F.interpolate(att_4d, size=out_hw, mode="bilinear", align_corners=False)
        return att_resized[0, 0].cpu().numpy()

    def save_rgb_no_title(img_hwc01, out_path):
        img_u8 = (np.clip(img_hwc01, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(out_path)

    def save_att_no_title(att_hw01, out_path):
        cmap_obj = plt.get_cmap(cmap)
        att_rgb = cmap_obj(np.clip(att_hw01, 0, 1))[..., :3]
        att_u8 = (att_rgb * 255).astype(np.uint8)
        Image.fromarray(att_u8).save(out_path)

    # ---------- sizes ----------
    T = cloudy_input.shape[0]
    H, W = cloudy_input.shape[-2], cloudy_input.shape[-1]
    out_hw = (H, W)

    # ---------- save inputs ----------
    if save_inputs_T is None:
        idxs = range(T)
    else:
        idxs = range(min(save_inputs_T, T))

    for i in idxs:
        img = linear_stretch(cloudy_input[i], percent=2)
        save_rgb_no_title(img, os.path.join(inputs_dir, f"{epoch_tag}_T{i+1}.png"))

    # ---------- save output / gt ----------
    out_img = linear_stretch(output, percent=2)
    gt_img  = linear_stretch(target, percent=2)

    save_rgb_no_title(out_img, os.path.join(output_dir, f"{epoch_tag}_output.png"))
    save_rgb_no_title(gt_img,  os.path.join(gt_dir,     f"{epoch_tag}_gt.png"))

    # ---------- save attention maps ----------
    if att_layer1 is not None:
        for i, att in enumerate(att_layer1):
            att_np = att_to_hw(att, out_hw)
            att01  = norm01(att_np)
            save_att_no_title(att01, os.path.join(att1_dir, f"{epoch_tag}_T{i+1}_att.png"))

    if att_layer2 is not None:
        for i, att in enumerate(att_layer2):
            att_np = att_to_hw(att, out_hw)
            att01  = norm01(att_np)
            save_att_no_title(att01, os.path.join(att2_dir, f"{epoch_tag}_T{i+1}_att.png"))

    print(f"[visualize_comparison] Saved images (no titles, no overlay) to: {save_dir} | epoch={epoch_tag}")
