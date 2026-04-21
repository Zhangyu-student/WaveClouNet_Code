# -*- coding: utf-8 -*-
"""
Fixed-parameter test/inference for your S2PatchDataset (13-band, 3 temporal inputs).

Run:
  python test_s2patch_fixed.py

It will:
  1) Inference on S2PatchDataset(mode="test")
  2) Save predicted 13-band TIF in ONE folder (SAVE_DIR), filename SAME as GT (e.g., GT__xxx.tif)
  3) Save visualization PNG (3x5) in SAME folder, filename based on GT (e.g., GT__xxx.png)

Visualization (3 rows x 5 cols):
  Row1: IN1 | IN2 | IN3 | PRED | GT
  Row2: T1 L1 Att | T2 L1 Att | T3 L1 Att | (blank) | (blank)
  Row3: T1 L2 Att | T2 L2 Att | T3 L2 Att | (blank) | (blank)

Model output supported:
  - pred
  - (pred, att1)
  - (pred, att1, att2)
  - {"pred": pred, "att1": att1, "att2": att2}  (or output/att_layer1/att_layer2 keys)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt

# =========================
# ✅ 0) 全部参数都写在这里：你只要改这一区域
# =========================
DATA_ROOT = r"M:\多时相云修复数据集\s2_asiaWest\S2PatchDataset-0117"
CKPT_PATH = r".\checkpoints\WaveDH_final_S2awest.pth"
SAVE_DIR  = r".\s2awest"   # ✅ 所有输出都到这个文件夹

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
NUM_WORKERS = 4
USE_SPLIT_SUBDIR = True    # 如果你的数据是 ROOT/test/ROIs... 则设 True（root = root / "test"）

SCALE = 10000.0
MEAN = None
STD  = None

# 你现在不需要导出输入/GT
SAVE_INPUTS_TIF = False
SAVE_GT_TIF     = False    # ⚠️ 因为 pred 的文件名和 GT 一样，保存 GT 会覆盖 pred（如需可改成 GT_REF__xxx.tif）

# 可视化用的 RGB 波段索引（0-based）
RGB_IDX = (3, 2, 1)
CLIP_LO = 2.0
CLIP_HI = 98.0

# 注意力图配色
ATT_CMAP = "viridis"

LOG_EVERY = 10


# =========================
# ✅ 1) Dataset
# =========================
from dataset import S2PatchDataset


# =========================
# ✅ 2) Build model
# =========================
def build_model() -> torch.nn.Module:
    from models.WaveCloudNet import WaveDH
    model = WaveDH(
        input_nc=10,
        output_nc=10,
        ngf=32,
        n_lo_b=2,
        n_bottles=3
    )
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str = "cuda") -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if isinstance(k, str) else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)


# =========================
# Utils
# =========================
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def resolve_patch_dir(patch_id: str) -> Path:
    """
    patch_id 是 dataset 返回的相对路径（相对于 root 或 root/test）
    """
    root = Path(DATA_ROOT)
    if USE_SPLIT_SUBDIR:
        root = root / "test"
    return root / patch_id


def find_single_file(patch_id: str, pattern: str) -> Path:
    pd = resolve_patch_dir(patch_id)
    files = sorted(pd.glob(pattern))
    if len(files) != 1:
        raise RuntimeError(f"Expect exactly 1 file for pattern={pattern} under {pd}, got {len(files)}")
    return files[0]


def ensure_chw(arr: np.ndarray) -> np.ndarray:
    """
    Accept:
      - [C,H,W]
      - [H,W,C]
      - [H,W]
    Return:
      - [C,H,W]
    """
    if arr.ndim == 2:
        return arr[None, ...]

    if arr.ndim != 3:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    # HWC: last dim looks like channels (<=20), first two dims look like spatial (>20)
    if arr.shape[-1] <= 20 and arr.shape[0] > 20 and arr.shape[1] > 20:
        return np.transpose(arr, (2, 0, 1))

    # CHW: first dim looks like channels (<=20), last two dims look like spatial (>20)
    if arr.shape[0] <= 20 and arr.shape[1] > 20 and arr.shape[2] > 20:
        return arr

    # fallback: treat as HWC
    return np.transpose(arr, (2, 0, 1))


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def make_unique_path(p: Path) -> Path:
    """
    如果文件名冲突，自动加 _dup001 / _dup002...
    """
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    for k in range(1, 999):
        cand = p.with_name(f"{stem}_dup{k:03d}{suf}")
        if not cand.exists():
            return cand
    return p.with_name(f"{stem}_dup999{suf}")


def save_tif_13band_hwc(
    save_path: str | Path,
    arr: np.ndarray,              # CHW or HWC
    scale: float = 10000.0,
    out_dtype=np.uint16,
    verify_readback: bool = True, # ✅ 写完读回检查
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    # --- to CHW ---
    chw = ensure_chw(arr).astype(np.float32)      # [C,H,W]
    C, H, W = chw.shape

    # --- scale back ---
    if out_dtype in (np.uint16, np.uint8, np.int16, np.int32):
        x = chw * float(scale) if scale != 1.0 else chw
        x = np.clip(x, 0, float(scale)).astype(out_dtype)
    else:
        x = chw.astype(np.float32)

    # --- write as HWC with OME metadata (multi-channel single image) ---
    hwc = np.transpose(x, (1, 2, 0))              # [H,W,C] => [256,256,13]

    tiff.imwrite(
        str(save_path),
        hwc,
        ome=True,                                 # ✅ 关键：OME-TIFF
        metadata={"axes": "YXS"},                 # ✅ 关键：Y,X,Samples
        photometric="minisblack",
        planarconfig="contig",                    # ✅ 通道按像素交错存储
    )

    # --- verify ---
    if verify_readback:
        rb = tiff.imread(str(save_path))
        if rb.shape != (H, W, C):
            raise RuntimeError(
                f"[TIF VERIFY FAIL] saved={save_path}\n"
                f"expect readback shape={(H,W,C)}, got {rb.shape}\n"
                f"Tip: some writers store as pages; OME should avoid that."
            )


def rgb_visualize_from_any(
    arr: np.ndarray,
    rgb_idx: Tuple[int, int, int] = (3, 2, 1),
    clip_percent: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    arr can be CHW or HWC, return uint8 RGB [H,W,3]
    """
    chw = ensure_chw(arr)
    C, H, W = chw.shape
    r, g, b = rgb_idx
    assert 0 <= r < C and 0 <= g < C and 0 <= b < C, f"rgb_idx out of range for C={C}"

    rgb = np.stack([chw[r], chw[g], chw[b]], axis=-1)  # [H,W,3]

    # per-channel percentile stretch (more stable than global)
    lo = np.percentile(rgb, clip_percent[0], axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, clip_percent[1], axis=(0, 1), keepdims=True)
    rgb = (rgb - lo) / (hi - lo + 1e-6)
    rgb = np.clip(rgb, 0.0, 1.0)

    return (rgb * 255.0 + 0.5).astype(np.uint8)


# =========================
# Attention visualization helpers
# =========================
def _to_tensor(att: Any) -> Optional[torch.Tensor]:
    if att is None:
        return None
    if torch.is_tensor(att):
        return att
    if isinstance(att, np.ndarray):
        return torch.from_numpy(att)
    return None


def _att_to_2d(att: Any) -> Optional[torch.Tensor]:
    """
    Convert various attention formats to a 2D tensor [H,W] (no batch).
    Accepts common shapes:
      - [H,W]
      - [1,H,W] / [C,H,W] -> take first channel
      - [H,W,1] -> squeeze
      - [B,1,H,W] / [B,C,H,W] -> take [0,0]
    """
    t = _to_tensor(att)
    if t is None:
        return None
    t = t.detach().float().cpu()

    if t.ndim == 2:
        return t

    if t.ndim == 3:
        # [C,H,W]
        if t.shape[0] <= 16 and t.shape[1] > 16 and t.shape[2] > 16:
            return t[0]
        # [H,W,C]
        if t.shape[-1] <= 16 and t.shape[0] > 16 and t.shape[1] > 16:
            return t[..., 0]
        return t[0]

    if t.ndim == 4:
        # [B,1,H,W] or [B,C,H,W]
        return t[0, 0]

    if t.ndim == 5:
        # [B,T,1,H,W] fallback
        return t[0, 0, 0]

    t = t.squeeze()
    if t.ndim == 2:
        return t
    if t.ndim >= 3:
        return t.reshape(t.shape[-2], t.shape[-1])
    return None


def _resize_norm_att(att2d: Optional[torch.Tensor], size_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Resize attention to (H,W) and normalize to 0..1 for visualization.
    """
    if att2d is None:
        return None
    a = att2d.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    a = F.interpolate(a, size=size_hw, mode="bilinear", align_corners=False)[0, 0]
    amin, amax = float(a.min()), float(a.max())
    if amax <= amin + 1e-12:
        return (a.numpy() * 0.0)
    a = (a - amin) / (amax - amin)
    return a.numpy()


def _get_temporal_att(att_pack: Any, t_index: int, size_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Extract t-th temporal attention map from different formats and return resized normalized [H,W] numpy.
    Supported:
      - list/tuple length>=3: each element is a map/tensor
      - torch tensor:
          [B,3,H,W] / [B,3,1,H,W] / [3,H,W] / [3,1,H,W]
      - dict with keys: att/attn/attention (then recurse)
    """
    if att_pack is None:
        return None

    if isinstance(att_pack, dict):
        for k in ("att", "attn", "attention"):
            if k in att_pack:
                return _get_temporal_att(att_pack[k], t_index, size_hw)
        return None

    if isinstance(att_pack, (list, tuple)):
        if len(att_pack) <= t_index:
            return None
        a2d = _att_to_2d(att_pack[t_index])
        return _resize_norm_att(a2d, size_hw)

    t = _to_tensor(att_pack)
    if t is None:
        return None
    t = t.detach().float().cpu()
    print(t.shape)
    # [B,3,1,H,W]
    if t.ndim == 5 and t.shape[1] == 3:
        a2d = _att_to_2d(t[0, t_index, 0])
        return _resize_norm_att(a2d, size_hw)

    # [B,3,H,W]
    if t.ndim == 4 and t.shape[1] == 3:
        a2d = _att_to_2d(t[0, t_index])
        return _resize_norm_att(a2d, size_hw)

    # [3,1,H,W]
    if t.ndim == 4 and t.shape[0] == 3:
        a2d = _att_to_2d(t[t_index, 0])
        return _resize_norm_att(a2d, size_hw)

    # [3,H,W]
    if t.ndim == 3 and t.shape[0] == 3:
        a2d = _att_to_2d(t[t_index])
        return _resize_norm_att(a2d, size_hw)

    # fallback: treat as single map
    a2d = _att_to_2d(t)
    return _resize_norm_att(a2d, size_hw)


def save_compare_png_with_att(
    save_path: str | Path,
    in1_rgb: np.ndarray,
    in2_rgb: np.ndarray,
    in3_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    title: str = "",
    att_layer1: Any = None,
    att_layer2: Any = None,
    att_cmap: str = "viridis",
) -> None:
    """
    3x5 layout:
      row1: IN1 IN2 IN3 PRED GT
      row2: T1 L1 Att, T2 L1 Att, T3 L1 Att, blank, blank
      row3: T1 L2 Att, T2 L2 Att, T3 L2 Att, blank, blank
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig = plt.figure(figsize=(20, 12))

    # ---- Row1 ----
    imgs = [in1_rgb, in2_rgb, in3_rgb, pred_rgb, gt_rgb]
    names = ["IN1", "IN2", "IN3", "PRED", "GT"]
    for i, (im, nm) in enumerate(zip(imgs, names)):
        ax = fig.add_subplot(3, 5, i + 1)
        ax.imshow(im)
        ax.set_title(nm, fontsize=10)
        ax.axis("off")

    Ht, Wt = in1_rgb.shape[0], in1_rgb.shape[1]
    size_hw = (Ht, Wt)

    # ---- Row2 (Layer1) ----
    if att_layer1 is not None:
        for ti in range(3):
            ax = fig.add_subplot(3, 5, 6 + ti)
            att_np = _get_temporal_att(att_layer1, ti, size_hw)
            if att_np is None:
                ax.axis("off")
                continue
            ax.imshow(att_np, cmap=att_cmap, vmin=0.0, vmax=1.0)
            ax.set_title(f"T{ti+1} L1 Att", fontsize=10)
            ax.axis("off")

    # ---- Row3 (Layer2) ----
    if att_layer2 is not None:
        for ti in range(3):
            ax = fig.add_subplot(3, 5, 11 + ti)
            att_np = _get_temporal_att(att_layer2, ti, size_hw)
            if att_np is None:
                ax.axis("off")
                continue
            ax.imshow(att_np, cmap=att_cmap, vmin=0.0, vmax=1.0)
            ax.set_title(f"T{ti+1} L2 Att", fontsize=10)
            ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Inference
# =========================
@torch.no_grad()
def main():
    ensure_dir(SAVE_DIR)

    ds = S2PatchDataset(
        data_root=DATA_ROOT,
        mode="test",
        use_split_subdir=USE_SPLIT_SUBDIR,
        scale=SCALE,
        mean=MEAN,
        std=STD,
        augment=False,
        strict=True,
        band_indices=[1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
    )
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_model().to(DEVICE)
    load_checkpoint(model, CKPT_PATH, device=DEVICE)
    model.eval()

    print(f"[INFO] device      = {DEVICE}")
    print(f"[INFO] samples     = {len(ds)}")
    print(f"[INFO] data_root   = {DATA_ROOT}")
    print(f"[INFO] ckpt        = {CKPT_PATH}")
    print(f"[INFO] save_dir    = {SAVE_DIR}")

    for bidx, batch in enumerate(loader):
        cond = batch["cond_image"].to(DEVICE)  # [B,3,C,H,W]
        gt   = batch["gt_image"].to(DEVICE)    # [B,C,H,W]
        paths = batch["path"]                  # list[str]

        pred, att1, att2 = model(cond)

        # ✅ 兜底：如果模型输出是 BHWC，则转成 BCHW
        if torch.is_tensor(pred) and pred.ndim == 4 and pred.shape[-1] <= 20 and pred.shape[1] > 20 and pred.shape[2] > 20:
            pred = pred.permute(0, 3, 1, 2).contiguous()

        # 兜底：有些模型输出 [B,1,C,H,W]
        if torch.is_tensor(pred) and pred.ndim == 5 and pred.shape[1] == 1:
            pred = pred[:, 0]

        assert torch.is_tensor(pred) and pred.ndim == 4, f"pred should be [B,C,H,W], got {type(pred)} {getattr(pred,'shape',None)}"

        B = pred.shape[0]
        for i in range(B):
            patch_id = str(paths[i]).replace("\\", "/")

            # ✅ 取 GT 的真实文件名（例如 GT__xxx.tif）
            gt_file = find_single_file(patch_id, "GT__*.tif")
            gt_name = gt_file.name
            base = Path(gt_name).stem  # GT__xxx

            # tensors -> numpy
            pred_arr = tensor_to_numpy(pred[i])
            gt_arr   = tensor_to_numpy(gt[i])
            in1_arr  = tensor_to_numpy(cond[i, 0])
            in2_arr  = tensor_to_numpy(cond[i, 1])
            in3_arr  = tensor_to_numpy(cond[i, 2])

            # ✅ 1) 保存 pred tif：文件名与 GT 一样（写到 SAVE_DIR）
            pred_tif_path = make_unique_path(Path(SAVE_DIR) / gt_name)
            save_tif_13band_hwc(pred_tif_path, pred_arr, scale=SCALE, out_dtype=np.uint16)

            # 可选：保存输入/GT（注意命名避免覆盖）
            if SAVE_INPUTS_TIF:
                suffix = base.split("GT__")[-1]
                save_tif_13band_hwc(Path(SAVE_DIR) / f"IN1__{suffix}.tif", in1_arr, scale=SCALE, out_dtype=np.uint16)
                save_tif_13band_hwc(Path(SAVE_DIR) / f"IN2__{suffix}.tif", in2_arr, scale=SCALE, out_dtype=np.uint16)
                save_tif_13band_hwc(Path(SAVE_DIR) / f"IN3__{suffix}.tif", in3_arr, scale=SCALE, out_dtype=np.uint16)

            if SAVE_GT_TIF:
                suffix = base.split("GT__")[-1]
                save_tif_13band_hwc(Path(SAVE_DIR) / f"GT_REF__{suffix}.tif", gt_arr, scale=SCALE, out_dtype=np.uint16)

            # ✅ 2) 保存对比 PNG（含注意力）
            in1_rgb  = rgb_visualize_from_any(in1_arr,  rgb_idx=RGB_IDX, clip_percent=(CLIP_LO, CLIP_HI))
            in2_rgb  = rgb_visualize_from_any(in2_arr,  rgb_idx=RGB_IDX, clip_percent=(CLIP_LO, CLIP_HI))
            in3_rgb  = rgb_visualize_from_any(in3_arr,  rgb_idx=RGB_IDX, clip_percent=(CLIP_LO, CLIP_HI))
            pred_rgb = rgb_visualize_from_any(pred_arr, rgb_idx=RGB_IDX, clip_percent=(CLIP_LO, CLIP_HI))
            gt_rgb   = rgb_visualize_from_any(gt_arr,   rgb_idx=RGB_IDX, clip_percent=(CLIP_LO, CLIP_HI))

            png_path = make_unique_path(Path(SAVE_DIR) / f"{base}.png")

            # 如果注意力是 batch 级别 tensor，这里取对应样本 i
            att1_i = att1[i] if torch.is_tensor(att1) and att1.ndim >= 1 else att1
            att2_i = att2[i] if torch.is_tensor(att2) and att2.ndim >= 1 else att2

            save_compare_png_with_att(
                png_path,
                in1_rgb, in2_rgb, in3_rgb, pred_rgb, gt_rgb,
                title=patch_id,
                att_layer1=att1_i,
                att_layer2=att2_i,
                att_cmap=ATT_CMAP
            )

        if (bidx + 1) % LOG_EVERY == 0:
            print(f"[INFO] processed batches: {bidx+1}/{len(loader)}")

    print("[DONE] Inference finished.")
    print(f"  - Outputs saved in ONE folder: {SAVE_DIR}")


if __name__ == "__main__":
    main()