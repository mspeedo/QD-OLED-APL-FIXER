# HDR-Native D4 Fast9 CNN Trainer

End-to-end pipeline to download free CC0 linear HDR images (or use your own 16-bit HDR10 PQ / `.exr` captures), resize & simulate realistic TAA/DLSS/motion blur strictly in **linear HDR light**, train a 56-FMA D4-invariant HDR sharpening CNN on your GPU (~2 minutes on an RTX 4080), and export drop-in HLSL for `RunModelA_Fast9()` in `Shaders/Neural_Sharpen_HDR.fx`.

## 1. Setup (Windows PowerShell / Linux)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: install CUDA-enabled PyTorch if pip installed CPU-only torch:
# pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

## 2. Download Free CC0 Linear HDR Images (Poly Haven)

Downloads 40 balanced 4K 32-bit `.exr` HDRIs (foliage/nature, urban/architecture, dark interiors, night/high-contrast) — no login required:

```powershell
python download_polyhaven_hdr.py --count 40 --res 4k --out dataset/raw_exr
```

*(Optional: you can also drop 16-bit HDR10 PQ `.png` images — such as [HDRTV1K](https://huggingface.co/datasets/chxy95/HDRTV1K) or Special K HDR screenshots — into `dataset/raw_pq/`.)*

## 3. Resize (2× Linear SSAA) & Generate TAA/DLSS Blurred Pairs

Downscales clean targets 2× in **linear HDR light** (alias-free 4× SSAA reference), applies realistic DLSS upscale + TAA blur + 10-bit PQ quantization in linear light, saves side-by-side visual previews to `dataset/preview/`, and extracts luminance- & contrast-stratified 3×3 patches:

```powershell
python prepare_hdr_dataset.py --exr-dir dataset/raw_exr --pq-dir dataset/raw_pq --out-dir dataset --downscale 2.0 --mode stationary
```

- Inspect `dataset/preview/*_blur_vs_gt.png` (Left = degraded input, Right = clean ground truth) to visually verify the blur strength.
- For a motion-pan blur dataset, run with `--mode motion --out-dir dataset_motion`.

## 4. Train & Export Drop-In HLSL

```powershell
python train_hdr_cnn.py --data dataset/patches_stationary.npz --epochs 60 --out exported_fast9_hlsl.txt
```

- Uses a 4-part HDR perceptual loss: Charbonnier PQ accuracy + **2.5× asymmetric OLED bright-halo penalty** + **linear-nit energy conservation** (prevents artificial foliage brightening) + **zero-DC pure-black & firefly suppression**.
- Paste the generated `RunModelA_Fast9(...)` function from `exported_fast9_hlsl.txt` directly into `Shaders/Neural_Sharpen_HDR.fx` (or export as `RunModelB_Fast9` with `--fn-name RunModelB_Fast9` and set `NEURAL_SHARPEN_MODEL=0` to A/B test live against Barbatos Model A in ReShade!).
