#!/usr/bin/env python3
"""
Resize linear .exr/.hdr or 16-bit HDR10 PQ .png images in LINEAR HDR light,
generate realistic TAA / DLSS / optional motion-blur degraded pairs, save visual
before/after previews, and extract a luminance- & contrast-stratified 3x3 patch
dataset (.npz) ready for train_hdr_cnn.py.

Examples:
    # 1. Process Poly Haven .exr files in dataset/raw_exr/
    python prepare_hdr_dataset.py --exr-dir dataset/raw_exr --out-dir dataset

    # 2. Also include 16-bit HDR10 PQ PNGs (e.g., HDRTV1K or Special K captures)
    python prepare_hdr_dataset.py --exr-dir dataset/raw_exr --pq-dir dataset/raw_pq --out-dir dataset

    # 3. Generate a motion-pan blur training set
    python prepare_hdr_dataset.py --exr-dir dataset/raw_exr --mode motion --out-dir dataset_motion
"""
import argparse
import glob
import math
import os
import sys

# Must be set before importing cv2 so OpenCV's built-in OpenEXR reader is enabled on Windows & Linux.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

# ST.2084 PQ constants (normalized linear [0, 1] where 1.0 == 10,000 nits)
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = (2523.0 / 4096.0) * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = (2413.0 / 4096.0) * 32.0
PQ_C3 = (2392.0 / 4096.0) * 32.0

LUMA_709  = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
LUMA_2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)


def linear_to_pq(lin: np.ndarray) -> np.ndarray:
    x = np.clip(lin, 0.0, 1.0).astype(np.float32)
    xm = np.power(x, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * xm) / (1.0 + PQ_C3 * xm), PQ_M2).astype(np.float32)


def pq_to_linear(pq: np.ndarray) -> np.ndarray:
    p = np.power(np.clip(pq, 0.0, 1.0).astype(np.float32), 1.0 / PQ_M2)
    num = np.maximum(p - PQ_C1, 0.0)
    den = np.maximum(PQ_C2 - PQ_C3 * p, 1e-12)
    return np.power(num / den, 1.0 / PQ_M1).astype(np.float32)


def load_linear_exr_luma(path: str, rng: np.random.Generator) -> np.ndarray | None:
    """Load linear .exr/.hdr image, crop equatorial band if 2:1 panorama, normalize exposure to realistic HDR nits."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  WARN: could not read {path}", file=sys.stderr)
        return None
    img = img.astype(np.float32)
    if img.ndim == 2:
        luma = np.maximum(img, 0.0)
    else:
        # OpenCV loads BGR
        rgb = np.maximum(img[..., :3][..., ::-1], 0.0)
        luma = rgb[..., 0] * LUMA_709[0] + rgb[..., 1] * LUMA_709[1] + rgb[..., 2] * LUMA_709[2]

    # Replace NaN/Inf safely
    luma = np.nan_to_num(luma, nan=0.0, posinf=1.0, neginf=0.0)

    h, w = luma.shape
    # If equirectangular 2:1 HDRI panorama, crop middle 64% vertically to avoid polar stretching
    if abs((w / max(h, 1)) - 2.0) < 0.15:
        y0 = int(h * 0.18)
        y1 = int(h * 0.82)
        luma = luma[y0:y1, :]

    # Normalize scene-referred exposure: map 75th percentile to realistic HDR mid-bright (~35-110 nits)
    pos = luma[luma > 1e-6]
    if pos.size < 1000:
        return None
    p75 = float(np.percentile(pos, 75))
    target_nits = float(rng.uniform(35.0, 115.0))
    scale = (target_nits / 10000.0) / max(p75, 1e-6)
    luma_norm = np.clip(luma * scale, 0.0, 1.0)
    return luma_norm.astype(np.float32)


def load_pq_image_luma(path: str) -> np.ndarray | None:
    """Load 16-bit (or 8-bit) HDR10 PQ image (.png/.tif) and convert to linear [0, 1] (1.0 == 10,000 nits)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  WARN: could not read {path}", file=sys.stderr)
        return None
    if img.dtype == np.uint16:
        pq = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        pq = img.astype(np.float32) / 255.0
    else:
        pq = np.clip(img.astype(np.float32), 0.0, 1.0)

    if pq.ndim == 2:
        pq_luma = pq
    else:
        rgb_pq = pq[..., :3][..., ::-1]
        # Convert PQ RGB -> linear RGB -> exact BT.2020 linear luminance
        rgb_lin = pq_to_linear(rgb_pq)
        luma_lin = rgb_lin[..., 0] * LUMA_2020[0] + rgb_lin[..., 1] * LUMA_2020[1] + rgb_lin[..., 2] * LUMA_2020[2]
        return np.clip(luma_lin, 0.0, 1.0).astype(np.float32)
    return pq_to_linear(pq_luma)


def make_motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    ksize = max(3, int(math.ceil(length * 2.0)) | 1)
    k = np.zeros((ksize, ksize), dtype=np.float32)
    c = ksize // 2
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    steps = max(16, ksize * 6)
    for t in np.linspace(-0.5 * length, 0.5 * length, steps):
        x = c + t * dx
        y = c + t * dy
        ix, iy = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - ix, y - iy
        for ox, wx in ((0, 1.0 - fx), (1, fx)):
            for oy, wy in ((0, 1.0 - fy), (1, fy)):
                px, py = ix + ox, iy + oy
                if 0 <= px < ksize and 0 <= py < ksize:
                    k[py, px] += wx * wy
    s = float(k.sum())
    return k / max(s, 1e-6)


def degrade_linear_hdr(gt_lin: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate realistic modern game TAA + DLSS/FSR (+ optional camera-pan motion) blur
    strictly in LINEAR HDR light, then encode to 10-bit-quantized ST.2084 PQ.
    """
    h, w = gt_lin.shape
    blur = gt_lin.copy()

    # 1. DLSS / FSR upscaling softness (downscale 0.58x..0.80x -> bicubic/bilinear back to 1.0x)
    if rng.random() < 0.70:
        scale = float(rng.uniform(0.58, 0.80))
        sw = max(32, int(round(w * scale)))
        sh = max(32, int(round(h * scale)))
        small = cv2.resize(blur, (sw, sh), interpolation=cv2.INTER_AREA)
        up_interp = cv2.INTER_CUBIC if rng.random() < 0.75 else cv2.INTER_LINEAR
        blur = cv2.resize(small, (w, h), interpolation=up_interp)
        blur = np.clip(blur, 0.0, 1.0)

    # 2. TAA temporal accumulation blur (mild isotropic Gaussian in linear HDR)
    sigma = float(rng.uniform(0.35, 0.85)) if mode == "stationary" else float(rng.uniform(0.50, 1.05))
    blur = cv2.GaussianBlur(blur, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    # 3. Directional motion-pan smear (for motion mode, or occasional mild pan in stationary mode)
    if mode == "motion" or rng.random() < 0.12:
        length = float(rng.uniform(1.3, 2.5)) if mode == "motion" else float(rng.uniform(0.7, 1.2))
        angle = float(rng.uniform(0.0, 180.0))
        mk = make_motion_kernel(length, angle)
        blur = cv2.filter2D(blur, -1, mk, borderType=cv2.BORDER_REPLICATE)

    blur = np.clip(blur, 0.0, 1.0)

    # Encode to ST.2084 PQ and simulate 10-bit PQ quantization + tiny shadow denoiser noise
    blur_pq = linear_to_pq(blur)
    if rng.random() < 0.80:
        blur_pq = np.round(blur_pq * 1023.0) / 1023.0
    # Subtle near-black noise (teaches the CNN NOT to sharpen shadow banding/DLSS denoiser grain)
    shadow_weight = np.clip((0.22 - blur_pq) * 5.0, 0.0, 1.0)
    noise = rng.normal(0.0, 0.00045, size=blur_pq.shape).astype(np.float32) * shadow_weight
    return np.clip(blur_pq + noise, 0.0, 1.0).astype(np.float32)


def save_preview(gt_pq: np.ndarray, blur_pq: np.ndarray, out_path: str) -> None:
    """Save a crisp 512x512 side-by-side crop (Left=Blurred Input, Right=Clean GT) for visual inspection."""
    h, w = gt_pq.shape
    ch, cw = min(512, h), min(512, w)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    # Simple PQ-to-sRGB-like preview visualization (linear nits / 300 nits -> gamma 2.2)
    def to_u8(pq_crop: np.ndarray) -> np.ndarray:
        lin = pq_to_linear(pq_crop) * (10000.0 / 250.0)
        tm = lin / (1.0 + lin)
        srgb = np.power(np.clip(tm, 0.0, 1.0), 1.0 / 2.2)
        return np.round(srgb * 255.0).astype(np.uint8)

    left = to_u8(blur_pq[y0:y0 + ch, x0:x0 + cw])
    right = to_u8(gt_pq[y0:y0 + ch, x0:x0 + cw])
    sep = np.full((ch, 4), 220, dtype=np.uint8)
    panel = np.concatenate([left, sep, right], axis=1)
    cv2.imwrite(out_path, panel)


def extract_stratified_patches(
    blur_pq: np.ndarray,
    gt_pq: np.ndarray,
    target_per_image: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 3x3 patches balanced across luminance (0..4000 nits) and local contrast (textures/foliage/edges)."""
    h, w = blur_pq.shape
    if h < 10 or w < 10:
        return np.empty((0, 9), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Vectorized 3x3 sliding view via slicing (zero giant intermediate copies)
    p0 = blur_pq[0:h - 2, 0:w - 2]
    p1 = blur_pq[0:h - 2, 1:w - 1]
    p2 = blur_pq[0:h - 2, 2:w]
    p3 = blur_pq[1:h - 1, 0:w - 2]
    p4 = blur_pq[1:h - 1, 1:w - 1]
    p5 = blur_pq[1:h - 1, 2:w]
    p6 = blur_pq[2:h,     0:w - 2]
    p7 = blur_pq[2:h,     1:w - 1]
    p8 = blur_pq[2:h,     2:w]
    y  = gt_pq[1:h - 1, 1:w - 1]

    pmax = np.maximum.reduce([p0, p1, p2, p3, p4, p5, p6, p7, p8])
    pmin = np.minimum.reduce([p0, p1, p2, p3, p4, p5, p6, p7, p8])
    rng3 = pmax - pmin

    # Stratified buckets: 4 luminance bands x 3 contrast bands (12 strata)
    lum_edges = [(0.005, 0.16), (0.16, 0.36), (0.36, 0.60), (0.60, 0.92)]
    con_edges = [(0.002, 0.03), (0.03, 0.12), (0.12, 1.00)]
    per_bin = max(32, target_per_image // (len(lum_edges) * len(con_edges)))

    xs, ys = [], []
    for lo_l, hi_l in lum_edges:
        m_l = (p4 >= lo_l) & (p4 < hi_l)
        for lo_c, hi_c in con_edges:
            mask = m_l & (rng3 >= lo_c) & (rng3 < hi_c)
            coords = np.flatnonzero(mask.ravel())
            if coords.size == 0:
                continue
            take = min(per_bin, coords.size)
            sel = rng.choice(coords, size=take, replace=False)
            patch = np.stack([
                p0.ravel()[sel], p1.ravel()[sel], p2.ravel()[sel],
                p3.ravel()[sel], p4.ravel()[sel], p5.ravel()[sel],
                p6.ravel()[sel], p7.ravel()[sel], p8.ravel()[sel],
            ], axis=1)
            xs.append(patch)
            ys.append(y.ravel()[sel])

    if not xs:
        return np.empty((0, 9), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)


def make_synthetic_guard_patches(n_flat: int, n_firefly: int, rng: np.random.Generator):
    """Explicit zero-DC flat/OLED-black patches and isolated 1-px specular fireflies with target delta == 0."""
    # Flat patches across 0.0 (exact OLED black) .. 0.95 PQ
    flat_vals = np.concatenate([
        np.zeros(n_flat // 4, dtype=np.float32),
        rng.uniform(0.0, 0.95, size=n_flat - n_flat // 4).astype(np.float32),
    ])
    x_flat = np.repeat(flat_vals[:, None], 9, axis=1)
    y_flat = flat_vals.copy()

    # Isolated 1-px bright/dark HDR specular fireflies (center spike 0.10..0.50 PQ above/below flat bg)
    bg = rng.uniform(0.05, 0.55, size=n_firefly).astype(np.float32)
    spike = rng.uniform(0.10, 0.45, size=n_firefly).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=n_firefly, p=[0.25, 0.75]).astype(np.float32)
    x_ff = np.repeat(bg[:, None], 9, axis=1)
    x_ff[:, 4] = np.clip(bg + sign * spike, 0.0, 0.98)
    y_ff = x_ff[:, 4].copy()

    return x_flat, y_flat, x_ff, y_ff


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize & degrade HDR images in linear light and extract training patches.")
    parser.add_argument("--exr-dir", default="dataset/raw_exr", help="Folder with linear .exr/.hdr images")
    parser.add_argument("--pq-dir", default="dataset/raw_pq", help="Optional folder with 16-bit HDR10 PQ .png/.tif images")
    parser.add_argument("--out-dir", default="dataset", help="Output folder for patches.npz and preview PNGs")
    parser.add_argument("--downscale", type=float, default=2.0, help="Linear SSAA downscale factor for clean GT (default: 2.0)")
    parser.add_argument("--mode", default="stationary", choices=["stationary", "motion"], help="Degradation mode")
    parser.add_argument("--variants", type=int, default=2, help="Degradation variants per image (default: 2)")
    parser.add_argument("--patches-per-variant", type=int, default=25000, help="Stratified patches per variant (default: 25000)")
    parser.add_argument("--previews", type=int, default=6, help="Number of side-by-side preview PNGs to save (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    sources = []
    if os.path.isdir(args.exr_dir):
        for ext in ("*.exr", "*.hdr", "*.EXR", "*.HDR"):
            for p in sorted(glob.glob(os.path.join(args.exr_dir, ext))):
                sources.append(("exr", p))
    if os.path.isdir(args.pq_dir):
        for ext in ("*.png", "*.tif", "*.tiff", "*.PNG"):
            for p in sorted(glob.glob(os.path.join(args.pq_dir, ext))):
                sources.append(("pq", p))

    if not sources:
        print(f"No HDR images found in '{args.exr_dir}' or '{args.pq_dir}'.", file=sys.stderr)
        print("Run `python download_polyhaven_hdr.py` first, or place .exr / 16-bit PQ .png files there.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(sources)} HDR images (mode={args.mode}, downscale={args.downscale}x, variants={args.variants})...")
    all_x, all_y = [], []
    previews_saved = 0

    for idx, (kind, path) in enumerate(sources, 1):
        gt_lin = load_linear_exr_luma(path, rng) if kind == "exr" else load_pq_image_luma(path)
        if gt_lin is None:
            continue

        # 2x area downscale in LINEAR light -> alias-free 4x SSAA optical ground truth
        if args.downscale > 1.01:
            h, w = gt_lin.shape
            nh = max(64, int(round(h / args.downscale)))
            nw = max(64, int(round(w / args.downscale)))
            gt_lin = cv2.resize(gt_lin, (nw, nh), interpolation=cv2.INTER_AREA)

        gt_pq = linear_to_pq(gt_lin)
        for v in range(args.variants):
            blur_pq = degrade_linear_hdr(gt_lin, args.mode, rng)
            if previews_saved < args.previews and v == 0:
                stem = os.path.splitext(os.path.basename(path))[0]
                save_preview(gt_pq, blur_pq, os.path.join(preview_dir, f"{stem}_blur_vs_gt.png"))
                previews_saved += 1

            px, py = extract_stratified_patches(blur_pq, gt_pq, args.patches_per_variant, rng)
            if px.shape[0] > 0:
                all_x.append(px)
                all_y.append(py)

        if idx % 5 == 0 or idx == len(sources):
            total_so_far = sum(a.shape[0] for a in all_x)
            print(f"  [{idx:3d}/{len(sources)}] {os.path.basename(path)} -> {total_so_far:,} patches collected")

    x_real = np.concatenate(all_x, axis=0)
    y_real = np.concatenate(all_y, axis=0)

    # Append synthetic zero-DC flat/OLED-black and specular firefly guard patches (~2% of dataset)
    n_guard = max(10000, x_real.shape[0] // 50)
    x_flat, y_flat, x_ff, y_ff = make_synthetic_guard_patches(n_guard, n_guard, rng)

    x_all = np.concatenate([x_real, x_flat, x_ff], axis=0)
    y_all = np.concatenate([y_real, y_flat, y_ff], axis=0)
    kind_all = np.concatenate([
        np.zeros(x_real.shape[0], dtype=np.uint8),       # 0 = real HDR patch
        np.ones(x_flat.shape[0], dtype=np.uint8),        # 1 = flat / pure black
        np.full(x_ff.shape[0], 2, dtype=np.uint8),       # 2 = specular firefly
    ])

    # Shuffle cleanly
    perm = rng.permutation(x_all.shape[0])
    x_all, y_all, kind_all = x_all[perm], y_all[perm], kind_all[perm]

    out_npz = os.path.join(args.out_dir, f"patches_{args.mode}.npz")
    np.savez(out_npz, x=x_all, y=y_all, kind=kind_all)
    print(f"\nSaved {x_all.shape[0]:,} stratified HDR patches -> {out_npz}")
    print(f"Saved {previews_saved} visual previews (Left=Degraded Input, Right=Clean GT) -> {preview_dir}")


if __name__ == "__main__":
    main()
