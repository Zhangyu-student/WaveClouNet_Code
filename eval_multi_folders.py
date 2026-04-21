import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from cleanfid import fid
from lpips import LPIPS
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# Edit this block if you prefer running the script directly in PyCharm.
CONFIG = {
    # "gt_dir": r"E:\beifen\Sen2_MTC_old实验结果\postprocess\GT_jpg",
    # "pred_dirs": [
    #     r"E:\beifen\Sen2_MTC_old实验结果\postprocess\Median_Filter",
    #     r"E:\beifen\Sen2_MTC_old实验结果\postprocess\pixel2pixel",
    #     r"E:\beifen\Sen2_MTC_old实验结果\postprocess\STGAN_Unet",
    #     r"E:\beifen\Sen2_MTC_old实验结果\postprocess\STGAN_ResNet",
    #     r"E:\beifen\Sen2_MTC_old实验结果\postprocess\CTGAN",
    #     r"E:\beifen\Sen2_MTC_old实验结果\postprocess\PMAA",
    #     r"E:\beifen\Sen2_MTC_old实验结果\postprocess\TMFNet",
    #     r"E:\beifen\Sen2_MTC_old实验结果\postprocess\WaveCloudNet",
    # ],
    "gt_dir": r"E:\beifen\Sen2_MTC实验结果\postprocess\GT",
    "pred_dirs": [
        r"E:\beifen\Sen2_MTC实验结果\postprocess\Median_Filter",
        r"E:\beifen\Sen2_MTC实验结果\postprocess\pixel2pixel",
        r"E:\beifen\Sen2_MTC实验结果\postprocess\STGAN_Unet",
        r"E:\beifen\Sen2_MTC实验结果\postprocess\STGAN_ResNet",
        r"E:\beifen\Sen2_MTC实验结果\postprocess\CTGAN",
        r"E:\beifen\Sen2_MTC实验结果\postprocess\PMAA",
        r"E:\beifen\Sen2_MTC实验结果\postprocess\TMFNet",
        r"E:\beifen\Sen2_MTC实验结果\postprocess\DiffCR",
        r"E:\beifen\Sen2_MTC实验结果\postprocess\WaveCloudNet",
    ],
    "output_dir": r".\evaluation_results_multi_folders_new",
    "lpips_net": "alex",
    "use_cpu": False,
    "match_mode": "exact",  # exact or contains
    "use_cli": False,
}


def list_images(image_dir):
    image_dir = Path(image_dir)
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def safe_name(path):
    resolved = str(Path(path).resolve())
    text = Path(path).name or "pred_dir"
    for ch in '<>:"/\\|?*':
        text = text.replace(ch, "_")
    digest = hashlib.md5(resolved.encode("utf-8")).hexdigest()[:8]
    return f"{text.strip('_')}_{digest}"


def build_gt_index(gt_dir):
    gt_paths = list_images(gt_dir)
    return {p.stem: p for p in gt_paths}, gt_paths


def match_gt(pred_path, gt_index, gt_paths, match_mode):
    if pred_path.stem in gt_index:
        return gt_index[pred_path.stem]

    if match_mode == "contains":
        pred_name = pred_path.stem
        for gt_path in gt_paths:
            if pred_name in gt_path.stem:
                return gt_path

    return None


def read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def calculate_sam_rgb(gt_rgb, pred_rgb):
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
    if not np.any(valid_mask):
        return float("nan")

    cos_theta = dot_product[valid_mask] / (norm_gt[valid_mask] * norm_pred[valid_mask])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return float(np.mean(np.degrees(np.arccos(cos_theta))))


def calculate_lpips(lpips_model, gt_img, pred_img, device):
    def numpy_to_torch(img):
        img_tensor = torch.from_numpy(img.astype(np.float32)) / 127.5 - 1.0
        return img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        return float(lpips_model(numpy_to_torch(gt_img), numpy_to_torch(pred_img)).item())


def prepare_fid_dir(rows, real_dir, pred_dir):
    for d in [real_dir, pred_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for row in rows:
        if row["status"] != "ok":
            continue
        filename = f"{row['filename']}.png"
        Image.fromarray(row["gt_img"]).save(real_dir / filename)
        Image.fromarray(row["pred_img"]).save(pred_dir / filename)


def nanmean(values):
    arr = np.array(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def evaluate_one_pred_dir(gt_dir, pred_dir, output_dir, gt_index, gt_paths, lpips_model, device, match_mode):
    pred_dir = Path(pred_dir)
    result_dir = output_dir / safe_name(pred_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pred_paths = list_images(pred_dir)

    for pred_path in tqdm(pred_paths, desc=f"Evaluating {pred_dir.name}"):
        gt_path = match_gt(pred_path, gt_index, gt_paths, match_mode)
        row = {
            "filename": pred_path.stem,
            "gt_path": str(gt_path) if gt_path else "",
            "pred_path": str(pred_path),
            "status": "ok" if gt_path else "missing_gt",
            "psnr": float("nan"),
            "ssim": float("nan"),
            "sam": float("nan"),
            "lpips": float("nan"),
        }

        if not gt_path:
            rows.append(row)
            continue

        try:
            gt_img = read_rgb(gt_path)
            pred_img = read_rgb(pred_path)

            if gt_img.shape[:2] != pred_img.shape[:2]:
                row["status"] = f"shape_mismatch_gt_{gt_img.shape}_pred_{pred_img.shape}"
                rows.append(row)
                continue

            row["psnr"] = float(psnr(gt_img, pred_img, data_range=255))
            row["ssim"] = float(ssim(gt_img, pred_img, data_range=255, channel_axis=-1))
            row["sam"] = calculate_sam_rgb(gt_img, pred_img)
            row["lpips"] = calculate_lpips(lpips_model, gt_img, pred_img, device)
            row["gt_img"] = gt_img
            row["pred_img"] = pred_img
        except Exception as exc:
            row["status"] = f"error: {exc}"

        rows.append(row)

    fid_real_dir = result_dir / "fid_real"
    fid_pred_dir = result_dir / "fid_pred"
    prepare_fid_dir(rows, fid_real_dir, fid_pred_dir)

    try:
        fid_value = float(fid.compute_fid(str(fid_real_dir), str(fid_pred_dir), device=device, num_workers=0))
    except Exception as exc:
        fid_value = float("nan")
        print(f"FID failed for {pred_dir}: {exc}")

    ok_rows = [r for r in rows if r["status"] == "ok"]
    summary = {
        "gt_dir": str(Path(gt_dir).resolve()),
        "pred_dir": str(pred_dir.resolve()),
        "output_dir": str(result_dir.resolve()),
        "total_pred_images": len(rows),
        "matched_images": len(ok_rows),
        "missing_or_failed_images": len(rows) - len(ok_rows),
        "psnr": nanmean([r["psnr"] for r in ok_rows]),
        "ssim": nanmean([r["ssim"] for r in ok_rows]),
        "sam": nanmean([r["sam"] for r in ok_rows]),
        "lpips": nanmean([r["lpips"] for r in ok_rows]),
        "fid": fid_value,
    }

    detail_path = result_dir / "detailed_metrics.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "gt_path", "pred_path", "status", "psnr", "ssim", "sam", "lpips"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    summary_path = result_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def evaluate_many(gt_dir, pred_dirs, output_dir, lpips_net="alex", use_cpu=False, match_mode="exact"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    print(f"Device: {device}")

    gt_index, gt_paths = build_gt_index(gt_dir)
    if not gt_paths:
        raise ValueError(f"No GT images found in {gt_dir}")

    lpips_model = LPIPS(net=lpips_net).to(device)
    lpips_model.eval()

    run_config = {
        "gt_dir": str(Path(gt_dir).resolve()),
        "pred_dirs": [str(Path(p).resolve()) for p in pred_dirs],
        "output_dir": str(output_dir.resolve()),
        "lpips_net": lpips_net,
        "use_cpu": use_cpu,
        "match_mode": match_mode,
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    summaries = []
    for pred_dir in pred_dirs:
        summaries.append(
            evaluate_one_pred_dir(
                gt_dir=gt_dir,
                pred_dir=pred_dir,
                output_dir=output_dir,
                gt_index=gt_index,
                gt_paths=gt_paths,
                lpips_model=lpips_model,
                device=device,
                match_mode=match_mode,
            )
        )

    summary_csv = output_dir / "all_summary_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "gt_dir",
            "pred_dir",
            "output_dir",
            "total_pred_images",
            "matched_images",
            "missing_or_failed_images",
            "psnr",
            "ssim",
            "sam",
            "lpips",
            "fid",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print("\nSummary")
    print("=" * 80)
    for item in summaries:
        print(f"Pred dir: {item['pred_dir']}")
        print(
            f"Matched: {item['matched_images']}/{item['total_pred_images']} | "
            f"PSNR: {item['psnr']:.4f} | SSIM: {item['ssim']:.4f} | "
            f"SAM: {item['sam']:.4f} | LPIPS: {item['lpips']:.4f} | FID: {item['fid']:.4f}"
        )
    print("=" * 80)
    print(f"All summaries saved to: {summary_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple restored-image folders against one GT folder.")
    parser.add_argument("--gt-dir", required=True, help="Ground-truth image directory.")
    parser.add_argument("--pred-dirs", nargs="+", required=True, help="One or more prediction directories.")
    parser.add_argument("--output-dir", default="evaluation_results_multi_folders", help="Output directory.")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--match-mode", default="exact", choices=["exact", "contains"])
    return parser.parse_args()


if __name__ == "__main__":
    if CONFIG.get("use_cli", False):
        args = parse_args()
        evaluate_many(
            gt_dir=args.gt_dir,
            pred_dirs=args.pred_dirs,
            output_dir=args.output_dir,
            lpips_net=args.lpips_net,
            use_cpu=args.use_cpu,
            match_mode=args.match_mode,
        )
    else:
        evaluate_many(
            gt_dir=CONFIG["gt_dir"],
            pred_dirs=CONFIG["pred_dirs"],
            output_dir=CONFIG["output_dir"],
            lpips_net=CONFIG["lpips_net"],
            use_cpu=CONFIG["use_cpu"],
            match_mode=CONFIG["match_mode"],
        )
