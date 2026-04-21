import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import random
import numpy as np
import tifffile as tiff
from torchvision.datasets import VisionDataset
import glob
from typing import Any, Callable, List, Optional, Tuple
import cv2
from base_dataset import get_params, get_transform

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(
            dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Sequence

import numpy as np
import torch
import tifffile as tiff
from torch.utils.data import Dataset

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:
    _HAS_PIL = False
    Image = None


class S2PatchDataset(Dataset):
    """
    适配结构：
      ROOT/
        ROIs1868/127/S2/patch_000003/
            GT__xxx.tif
            IN1__xxx.tif
            IN2__xxx.tif
            IN3__xxx.tif
            (可选) GT__xxx.png / IN1__xxx.png ...
            (可选) META__mode=...txt

    返回：
      ret['gt_image']   : Tensor [C,H,W]
      ret['cond_image'] : Tensor [3,C,H,W]   # 3个时相输入
      ret['path']       : str (用 patch 相对路径做标识)
    """

    def __init__(
            self,
            data_root: str,
            mode: str = "train",
            use_split_subdir: bool = False,  # 如果你最终做成 ROOT/train/... 就设 True
            min_frames_per_patch: int = 4,
            scale: float = 10000.0,  # Sentinel 常用 0-10000，若你已归一化可设 1.0
            mean: Optional[List[float]] = None,
            std: Optional[List[float]] = None,
            augment: bool = True,
            strict: bool = True,  # True: 找不到 1+3 就跳过；False: 尽量凑
            seed: int = 1234,
            band_indices: Optional[Sequence[int]] = None,  # ✅ 新增：选择返回的波段索引（0-based，可为负）
    ):
        self.data_root = data_root
        self.mode = mode
        self.use_split_subdir = use_split_subdir
        self.min_frames_per_patch = min_frames_per_patch
        self.scale = float(scale)
        self.mean = mean
        self.std = std
        self.augment = augment and (mode == "train")
        self.strict = strict
        self.rng = np.random.RandomState(seed)
        self.band_indices = None if band_indices is None else list(band_indices)

        root = Path(data_root)
        if use_split_subdir:
            root = root / mode  # e.g. ROOT/train/ROIs...
        if not root.exists():
            raise FileNotFoundError(f"data root not found: {root}")

        # 如果提供了 mean/std，建议长度与选择的 band 对齐
        if self.band_indices is not None and (self.mean is not None and self.std is not None):
            if len(self.mean) != len(self.band_indices) or len(self.std) != len(self.band_indices):
                raise ValueError(
                    f"mean/std length must match selected bands: "
                    f"len(mean)={len(self.mean)}, len(std)={len(self.std)}, bands={len(self.band_indices)}"
                )

        # 扫描所有 patch_* 文件夹
        patch_dirs = sorted([p for p in root.rglob("patch_*") if p.is_dir()])

        self.samples: List[Dict[str, str]] = []
        for pd in patch_dirs:
            # 只拿 tif（忽略 png）
            gt = sorted(pd.glob("GT__*.tif"))
            in1 = sorted(pd.glob("IN1__*.tif"))
            in2 = sorted(pd.glob("IN2__*.tif"))
            in3 = sorted(pd.glob("IN3__*.tif"))

            if len(gt) == 1 and len(in1) == 1 and len(in2) == 1 and len(in3) == 1:
                self.samples.append({
                    "gt": str(gt[0]),
                    "in1": str(in1[0]),
                    "in2": str(in2[0]),
                    "in3": str(in3[0]),
                    "id": str(pd.relative_to(root)).replace("\\", "/"),
                })
            else:
                if strict:
                    continue
                # 非严格：尽量凑（比如同一前缀多个文件时取第一个）
                if len(gt) == 0 or (len(in1) + len(in2) + len(in3)) == 0:
                    continue
                # 兜底：从所有 IN*.tif 里按名字排序取前三个
                all_in = sorted(pd.glob("IN*__*.tif"))
                if len(all_in) < 3:
                    continue
                self.samples.append({
                    "gt": str(gt[0]),
                    "in1": str(all_in[0]),
                    "in2": str(all_in[1]),
                    "in3": str(all_in[2]),
                    "id": str(pd.relative_to(root)).replace("\\", "/"),
                })

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under: {root}")

        # 训练增强参数（保证同一序列一致）
        if self.augment:
            n = len(self.samples)
            self.rot_k = self.rng.randint(0, 4, n)  # 0/1/2/3 => 0/90/180/270
            self.flip_k = self.rng.randint(0, 3, n)  # 0 none, 1 hflip, 2 vflip

        print(f"[S2PatchDataset] Loaded {len(self.samples)} samples for mode={mode} (root={root})")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_chw(img: np.ndarray) -> np.ndarray:
        """
        tifffile 可能返回 (H,W,C) 或 (C,H,W) 或 (H,W)
        统一转成 (C,H,W)
        """
        if img.ndim == 2:
            img = img[None, ...]  # [1,H,W]
        elif img.ndim == 3:
            # 若第0维很小且像通道数，认为是 CHW；否则认为是 HWC
            if img.shape[0] <= 20 and img.shape[1] > 20 and img.shape[2] > 20:
                # CHW
                pass
            else:
                # HWC -> CHW
                img = np.transpose(img, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported tif shape: {img.shape}")
        return img

    def _augment_chw(self, img: np.ndarray, idx: int) -> np.ndarray:
        if not self.augment:
            return img
        # img: [C,H,W]
        fk = int(self.flip_k[idx])
        rk = int(self.rot_k[idx])

        # hflip：沿 W 翻转；vflip：沿 H 翻转
        if fk == 1:
            img = img[:, :, ::-1]
        elif fk == 2:
            img = img[:, ::-1, :]

        if rk != 0:
            img = np.rot90(img, k=rk, axes=(1, 2)).copy()
        return img

    def _select_bands(self, img_chw: np.ndarray) -> np.ndarray:
        """
        img_chw: [C,H,W] -> 选出指定波段（0-based，可为负）
        """
        if self.band_indices is None:
            return img_chw
        C = img_chw.shape[0]
        idx = np.array(self.band_indices, dtype=int)
        idx = np.where(idx < 0, idx + C, idx)  # 支持负索引

        if np.any(idx < 0) or np.any(idx >= C):
            raise IndexError(f"band_indices out of range: {self.band_indices}, but C={C}")
        return img_chw[idx, :, :]

    def image_read(self, image_path: str, idx: int) -> torch.Tensor:
        img = tiff.imread(image_path)
        img = self._to_chw(img).astype(np.float32)

        # 数据增强（对 GT / IN 保持一致）
        img = self._augment_chw(img, idx)

        # ✅ 选择波段（核心）
        img = self._select_bands(img)

        # 归一化
        if self.scale != 1.0:
            img = img / self.scale

        x = torch.from_numpy(img).float()  # [C,H,W]

        # 标准化（可选）
        if self.mean is not None and self.std is not None:
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
            x = (x - mean) / (std + 1e-12)

        return x

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        s = self.samples[index]
        in_paths = [s["in1"], s["in2"], s["in3"]]
        gt_path = s["gt"]

        cloud_imgs = [self.image_read(p, index) for p in in_paths]  # 3 * [C,H,W]
        gt_img = self.image_read(gt_path, index)  # [C,H,W]

        cond = torch.stack(cloud_imgs, dim=0)  # [3,C,H,W]

        return {
            "gt_image": gt_img,
            "cond_image": cond,
            "path": s["id"],
        }


class S2PatchRGBDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            mode: str = "train",
            use_split_subdir: bool = False,  # if ROOT/train/... then True
            augment: bool = True,
            strict: bool = True,  # True: missing any png => skip sample
            seed: int = 1234,
    ):
        self.mode = mode
        self.augment = bool(augment and (mode == "train"))
        self.strict = strict
        self.rng = np.random.RandomState(seed)

        root = Path(data_root)
        if use_split_subdir:
            root = root / mode
        if not root.exists():
            raise FileNotFoundError(f"data root not found: {root}")

        # scan patch dirs
        patch_dirs = sorted([p for p in root.rglob("patch_*") if p.is_dir()])

        def tif_to_png(tif_path: Path) -> Path:
            # same name, just change suffix
            return tif_path.with_suffix(".png")

        self.samples: List[Dict[str, str]] = []
        for pd in patch_dirs:
            gt = sorted(pd.glob("GT__*.tif"))
            in1 = sorted(pd.glob("IN1__*.tif"))
            in2 = sorted(pd.glob("IN2__*.tif"))
            in3 = sorted(pd.glob("IN3__*.tif"))

            if not (len(gt) == len(in1) == len(in2) == len(in3) == 1):
                if strict:
                    continue
                # non-strict: try to take first GT and first 3 IN*
                if len(gt) == 0:
                    continue
                all_in = sorted(pd.glob("IN*__*.tif"))
                if len(all_in) < 3:
                    continue
                gt_tif, in1_tif, in2_tif, in3_tif = gt[0], all_in[0], all_in[1], all_in[2]
            else:
                gt_tif, in1_tif, in2_tif, in3_tif = gt[0], in1[0], in2[0], in3[0]

            # find pngs by tif name
            gt_png = tif_to_png(gt_tif)
            in1_png = tif_to_png(in1_tif)
            in2_png = tif_to_png(in2_tif)
            in3_png = tif_to_png(in3_tif)

            if self.strict:
                if not (gt_png.exists() and in1_png.exists() and in2_png.exists() and in3_png.exists()):
                    continue

            self.samples.append({
                "gt_png": str(gt_png),
                "in1_png": str(in1_png),
                "in2_png": str(in2_png),
                "in3_png": str(in3_png),
                "id": str(pd.relative_to(root)).replace("\\", "/"),
            })

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid RGB samples found under: {root}. "
                               f"Check PNG existence and naming consistency with TIF.")

        # deterministic per-sample augmentation parameters
        if self.augment:
            n = len(self.samples)
            self.rot_k = self.rng.randint(0, 4, n)  # 0/1/2/3 => 0/90/180/270
            self.flip_k = self.rng.randint(0, 3, n)  # 0 none, 1 hflip, 2 vflip

        print(f"[S2PatchRGBDataset] Loaded {len(self.samples)} samples for mode={mode} (root={root})")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_chw(self, img: np.ndarray, idx: int) -> np.ndarray:
        if not self.augment:
            return img
        fk = int(self.flip_k[idx])
        rk = int(self.rot_k[idx])

        if fk == 1:
            img = img[:, :, ::-1]
        elif fk == 2:
            img = img[:, ::-1, :]

        if rk != 0:
            img = np.rot90(img, k=rk, axes=(1, 2))

        return img.copy()

    def _read_png_rgb01(self, path: str) -> np.ndarray:
        if _HAS_CV2:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"png not found/unreadable: {path}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # HWC
        else:
            if not _HAS_PIL:
                raise RuntimeError("Need cv2 or PIL to read PNG.")
            im = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0  # HWC

        im = np.transpose(im, (2, 0, 1))  # CHW: [3,H,W]
        return im

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]

        in_paths = [s["in1_png"], s["in2_png"], s["in3_png"]]
        gt_path = s["gt_png"]

        cloud_imgs = [self._read_png_rgb01(p) for p in in_paths]  # 3 * [3,H,W]
        gt_img = self._read_png_rgb01(gt_path)  # [3,H,W]

        # consistent augmentation across the 4 images
        cloud_imgs = [self._augment_chw(im, index) for im in cloud_imgs]
        gt_img = self._augment_chw(gt_img, index)

        cond = torch.from_numpy(np.stack(cloud_imgs, axis=0).copy()).float()  # [3,3,H,W]
        gt = torch.from_numpy(gt_img.copy()).float()  # [3,H,W]

        return {
            "gt_image": gt,
            "cond_image": cond,
            "path": s["id"],
        }


class Landsat(data.Dataset):
    """
    适用于最终划分好的多时相数据集的 PyTorch Dataset 类。

    结构：FINAL_DATASET_ROOT/mode/clear/file.tif & FINAL_DATASET_ROOT/mode/cloudy/file.tif
    """

    SUBFOLDERS = ("cloudy", "clear")

    # 假设您的文件格式是 [R_C]_[SeqXXX]_[T/GT]_[DATE].tif

    def __init__(self, data_root: str, mode: str = 'train'):
        # ... (初始化部分保持不变)
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []
        self.index = 0

        # 1. 构造基础路径
        split_root = os.path.join(self.data_root, self.mode)
        cloudy_path = os.path.join(split_root, self.SUBFOLDERS[0])
        clear_path = os.path.join(split_root, self.SUBFOLDERS[1])

        if not os.path.isdir(cloudy_path) or not os.path.isdir(clear_path):
            raise FileNotFoundError(f"无法在 {split_root} 中找到 'cloudy' 或 'clear' 文件夹。")

        # 2. 收集所有序列ID (仍然基于 T1 文件，以确保序列的起点是明确的)
        t1_paths = sorted(glob.glob(os.path.join(cloudy_path, '*_T1_*.tif')))

        # 3. 构建文件对 (Sequence Matching)
        for t1_path in t1_paths:
            # 提取序列前缀 [R_C]_[SeqXXX]
            filename = os.path.basename(t1_path)
            # 通过 split('_')[:3] 提取 R_C_SeqXXX
            base_prefix = "_".join(filename.split('_')[:3])

            # T1 文件已经有了 (t1_path)，但为了逻辑一致性，也通过 glob 确认一次
            # 查找 T1 文件 (理论上只有 t1_path 自己)
            t1_paths_found = glob.glob(os.path.join(cloudy_path, f"{base_prefix}_T1_*.tif"))

            # 查找 T2 文件
            t2_paths = glob.glob(os.path.join(cloudy_path, f"{base_prefix}_T2_*.tif"))

            # 查找 T3 文件
            t3_paths = glob.glob(os.path.join(cloudy_path, f"{base_prefix}_T3_*.tif"))

            # 查找对应的 Clear (GT) 文件
            gt_filename_pattern = os.path.join(clear_path, f"{base_prefix}_GT_*.tif")
            gt_paths = glob.glob(gt_filename_pattern)

            # 检查匹配结果
            if len(t1_paths_found) == 1 and len(t2_paths) == 1 and len(t3_paths) == 1 and len(gt_paths) == 1:
                # 成功匹配所有 4 个文件
                image_cloud_path0 = t1_paths_found[0]  # T1
                image_cloud_path1 = t2_paths[0]  # T2
                image_cloud_path2 = t3_paths[0]  # T3
                image_cloudless_path = gt_paths[0]  # GT

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(base_prefix)

            else:
                # 任何一个时相或 GT 缺失或重复，则警告并跳过
                missing_info = []
                if len(t1_paths_found) != 1: missing_info.append(f"T1: {len(t1_paths_found)} 个")
                if len(t2_paths) != 1: missing_info.append(f"T2: {len(t2_paths)} 个")
                if len(t3_paths) != 1: missing_info.append(f"T3: {len(t3_paths)} 个")
                if len(gt_paths) != 1: missing_info.append(f"GT: {len(gt_paths)} 个")

                print(f"⚠️ 警告: 序列 {base_prefix} 文件不完整或重复 ({', '.join(missing_info)})，跳过。")

        # 4. 训练模式下的数据增强参数 (保持不变)
        if self.mode == 'train':
            num_sequences = len(self.filepair)
            self.augment_rotation_param = np.random.randint(0, 4, num_sequences)
            self.augment_flip_param = np.random.randint(0, 3, num_sequences)
        print(f"成功加载 {len(self.filepair)} 个 {self.mode} 序列。")

    def __getitem__(self, index: int) -> dict:
        """
        返回一个字典，包含 cond_image (3张云图), gt_image (1张真值图), path。
        """
        # 获取当前序列的所有文件路径
        cloud_image_paths = self.filepair[index][:3]
        cloudless_image_path = self.filepair[index][3]

        # 1. 读取和增强所有图像
        image_cloud_tensors = []
        for t_index, path in enumerate(cloud_image_paths):
            # 将索引和当前时相的索引传入，用于读取和增强
            img_tensor = self.image_read(path, index)
            image_cloud_tensors.append(img_tensor)

        image_cloudless = self.image_read(cloudless_image_path, index)

        # 2. 构造输出字典
        ret = {}
        # GT 图像 (7 个波段)
        ret['gt_image'] = image_cloudless

        # 条件图像 (3个时相，每个 7 个波段)
        ret['cond_image'] = torch.cat([img.unsqueeze(0) for img in image_cloud_tensors], dim=0)

        ret['path'] = self.image_name[index] + ".tif"  # 使用序列前缀作为路径标识
        return ret

    def __len__(self) -> int:
        return len(self.filepair)

    def image_read(self, image_path: str, seq_index: int) -> torch.Tensor:
        """
        读取 TIF 文件，进行归一化和标准化，并在训练模式下进行增强。
        """
        # 读取 TIF 文件 (假设 TIF 文件为 C, H, W 格式)
        img = tiff.imread(image_path)

        # 原始数据范围 [0, 10000+]，转换为 (C, H, W)
        # 假设 tifffile.imread 返回的是 (H, W, C) 或 (C, H, W)。如果返回 (H, W, C)，需要转置
        if img.shape[0] != 7:
            img = img.transpose((2, 0, 1))  # 从 (H, W, C) -> (C, H, W)

        # 数据增强 (仅在训练模式下)
        if self.mode == 'train':
            # 使用对应序列的随机参数
            flip_param = self.augment_flip_param[seq_index]
            rot_param = self.augment_rotation_param[seq_index]

            if not flip_param == 0:
                # 1: 左右翻转 (dim 2), 2: 上下翻转 (dim 1)
                img = np.flip(img, flip_param)
            if not rot_param == 0:
                # 旋转 (dim 1, 2)
                img = np.rot90(img, rot_param, (1, 2))

        # 转换为 PyTorch Tensor (float32)
        image = torch.from_numpy(img.copy()).float()

        # 1. 归一化 (除以 10000.0)
        image = image / 10000.0

        # 2. 标准化 (减均值 0.5, 除方差 0.5)
        # 假设所有 7 个波段都使用相同的均值和标准差
        mean = torch.as_tensor([0.5] * 7, dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5] * 7, dtype=image.dtype, device=image.device)

        # 调整形状以匹配 (C, H, W)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

        image.sub_(mean).div_(std)

        return image


class Sen2_MTC_New_Multi(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []

        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(
                self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'test.txt'), dtype=str)
        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless'))]

            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[
            index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]
        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        # return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, self.image_name[index]
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        ret['cond_image'] = torch.cat([image_cloud0[:3, :, :].unsqueeze(0), image_cloud1[:3, :, :].unsqueeze(0),
                                       image_cloud2[:3, :, :].unsqueeze(0)], dim=0)
        ret['path'] = self.image_name[index] + ".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))

        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index // 4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1

        if self.index // 4 >= len(self.filepair):
            self.index = 0

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image


class MultipleDataset(VisionDataset):
    SUBFOLDERS = ("cloudy", "clear")

    def __init__(
            self,
            data_root: str,
            band: int = 4,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(data_root, transform, target_transform, transforms)
        root = data_root
        self.band = band

        cloudy_0_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_0.jpg")
        cloudy_1_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_1.jpg")
        cloudy_2_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_2.jpg")

        cloudy_0_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_0_ir.jpg")
        cloudy_1_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_1_ir.jpg")
        cloudy_2_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_2_ir.jpg")

        clear_pathname = os.path.join(root, self.SUBFOLDERS[1], "*")

        self.cloudy_0_rgb_paths = sorted(glob.glob(cloudy_0_rgb_pathname))
        self.cloudy_1_rgb_paths = sorted(glob.glob(cloudy_1_rgb_pathname))
        self.cloudy_2_rgb_paths = sorted(glob.glob(cloudy_2_rgb_pathname))

        self.cloudy_0_ir_paths = sorted(glob.glob(cloudy_0_ir_pathname))
        self.cloudy_1_ir_paths = sorted(glob.glob(cloudy_1_ir_pathname))
        self.cloudy_2_ir_paths = sorted(glob.glob(cloudy_2_ir_pathname))

        self.clear_paths = sorted(glob.glob(clear_pathname))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        cloudy_0_rgb = Image.open(
            self.cloudy_0_rgb_paths[index]).convert('RGB')
        cloudy_1_rgb = Image.open(
            self.cloudy_1_rgb_paths[index]).convert('RGB')
        cloudy_2_rgb = Image.open(
            self.cloudy_2_rgb_paths[index]).convert('RGB')

        if self.band == 4:
            cloudy_0_ir = Image.open(
                self.cloudy_0_ir_paths[index]).convert('RGB')
            cloudy_1_ir = Image.open(
                self.cloudy_1_ir_paths[index]).convert('RGB')
            cloudy_2_ir = Image.open(
                self.cloudy_2_ir_paths[index]).convert('RGB')
        else:
            pass

        clear = Image.open(self.clear_paths[index]).convert('RGB')

        params = get_params(size=clear.size)
        transform_params = get_transform(self.band, params)

        cloudy_0_rgb_tensor = transform_params(cloudy_0_rgb)
        cloudy_1_rgb_tensor = transform_params(cloudy_1_rgb)
        cloudy_2_rgb_tensor = transform_params(cloudy_2_rgb)
        if self.band == 4:
            cloudy_0_ir_tensor = transform_params(cloudy_0_ir)[:1, ...]
            cloudy_1_ir_tensor = transform_params(cloudy_1_ir)[:1, ...]
            cloudy_2_ir_tensor = transform_params(cloudy_2_ir)[:1, ...]
        else:
            cloudy_0_ir_tensor = None
            cloudy_1_ir_tensor = None
            cloudy_2_ir_tensor = None
        clear_tensor = transform_params(clear)

        cloudy_0 = torch.cat(
            [i for i in [cloudy_0_rgb_tensor, cloudy_0_ir_tensor] if i is not None])
        cloudy_1 = torch.cat(
            [i for i in [cloudy_1_rgb_tensor, cloudy_1_ir_tensor] if i is not None])
        cloudy_2 = torch.cat(
            [i for i in [cloudy_2_rgb_tensor, cloudy_2_ir_tensor] if i is not None])

        clear = clear_tensor

        ret = {}
        ret['gt_image'] = clear
        ret['cond_image'] = torch.cat([cloudy_0.unsqueeze(0), cloudy_1.unsqueeze(0),
                                       cloudy_2.unsqueeze(0)], dim=0)
        ret['path'] = self.clear_paths[index].replace("jpg", 'png')

        return ret

    def __len__(self) -> int:
        print("len(total_data) =", len(self.clear_paths))
        return len(self.clear_paths)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for item in MultipleDataset(data_root=r"D:\研究生\论文代码\dataset\multipleImage", band=3):
        print(item["cond_image"].shape, item["cond_image"].dtype)
        print(item["gt_image"].shape, item["gt_image"].dtype)
        print(item["path"].replace("\\", "/"))

        plt.figure(figsize=(8, 32), dpi=300)

        plt.subplot(1, 4, 1)
        plt.title("cond_image_0")
        plt.imshow(item["cond_image"][0, ...].permute(1, 2, 0) * 0.5 + 0.5)

        plt.subplot(1, 4, 2)
        plt.title("cond_image_1")
        plt.imshow(item["cond_image"][1, ...].permute(1, 2, 0) * 0.5 + 0.5)

        plt.subplot(1, 4, 3)
        plt.title("cond_image_2")
        plt.imshow(item["cond_image"][2, ...].permute(1, 2, 0) * 0.5 + 0.5)

        plt.subplot(1, 4, 4)
        plt.title("gt_image")
        plt.imshow(item["gt_image"].permute(1, 2, 0) * 0.5 + 0.5)

        plt.savefig("paired.png", bbox_inches="tight")

        break
