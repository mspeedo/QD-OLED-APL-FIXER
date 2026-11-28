#!/usr/bin/env python3
"""
Download a balanced, CC0-licensed set of real-world 32-bit float linear .exr HDRIs
from Poly Haven's free public API (https://polyhaven.com/our-api).

No API key or account required. Uses Python stdlib only.
All downloaded assets are CC0 public domain (credit: Poly Haven).

Example:
    python download_polyhaven_hdr.py --count 40 --res 4k --out dataset/raw_exr
"""
import argparse
import json
import os
import sys
import urllib.request

API_BASE = "https://api.polyhaven.com"
USER_AGENT = "QD-OLED-APL-FIXER-HDR-Trainer/1.0 (https://github.com/mspeedo/QD-OLED-APL-FIXER)"


def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dst: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dst + ".part"
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)


def pick_balanced_assets(assets: dict, count: int) -> list[tuple[str, str]]:
    """Select an even mix of nature/foliage, urban, indoor, and night/high-contrast HDR scenes."""
    buckets = {
        "nature_foliage": [],
        "urban_arch": [],
        "indoor_dark": [],
        "night_highcontrast": [],
        "other": [],
    }
    for asset_id, meta in sorted(assets.items()):
        cats = set(c.lower() for c in meta.get("categories", []))
        tags = set(t.lower() for t in meta.get("tags", []))
        all_labels = cats | tags

        if "night" in cats or "sunrise-sunset" in cats or "high contrast" in cats:
            buckets["night_highcontrast"].append(asset_id)
        elif "nature" in cats or "forest" in all_labels or "trees" in all_labels or "grass" in all_labels:
            buckets["nature_foliage"].append(asset_id)
        elif "urban" in cats or "architecture" in all_labels or "street" in all_labels:
            buckets["urban_arch"].append(asset_id)
        elif "indoor" in cats or "studio" in cats:
            buckets["indoor_dark"].append(asset_id)
        else:
            buckets["other"].append(asset_id)

    order = ["nature_foliage", "urban_arch", "indoor_dark", "night_highcontrast", "other"]
    # Emphasize foliage/nature (2 slots per round) because fine distant foliage is hardest to sharpen cleanly.
    round_pattern = ["nature_foliage", "nature_foliage", "urban_arch", "indoor_dark", "night_highcontrast", "other"]
    idxs = {k: 0 for k in order}
    picked = []
    seen = set()

    while len(picked) < count:
        progressed = False
        for b in round_pattern:
            if len(picked) >= count:
                break
            arr = buckets[b]
            while idxs[b] < len(arr) and arr[idxs[b]] in seen:
                idxs[b] += 1
            if idxs[b] < len(arr):
                aid = arr[idxs[b]]
                idxs[b] += 1
                seen.add(aid)
                picked.append((aid, b))
                progressed = True
        if not progressed:
            break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CC0 linear .exr HDRIs from Poly Haven.")
    parser.add_argument("--count", type=int, default=40, help="Number of .exr HDRIs to download (default: 40)")
    parser.add_argument("--res", default="4k", choices=["2k", "4k", "8k"], help="HDRI resolution (default: 4k)")
    parser.add_argument("--out", default="dataset/raw_exr", help="Output directory (default: dataset/raw_exr)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Fetching HDRI catalog from {API_BASE}/assets?t=hdris ...")
    assets = http_get_json(f"{API_BASE}/assets?t=hdris")
    selected = pick_balanced_assets(assets, args.count)
    print(f"Selected {len(selected)} balanced CC0 HDRIs at {args.res} (.exr) -> {args.out}")

    ok = 0
    for i, (aid, bucket) in enumerate(selected, 1):
        dst = os.path.join(args.out, f"{aid}_{args.res}.exr")
        if os.path.exists(dst) and os.path.getsize(dst) > 1024:
            print(f"  [{i:2d}/{len(selected)}] SKIP (exists): {os.path.basename(dst)} [{bucket}]")
            ok += 1
            continue
        try:
            files = http_get_json(f"{API_BASE}/files/{aid}")
            exr_info = files.get("hdri", {}).get(args.res, {}).get("exr")
            if not exr_info or "url" not in exr_info:
                print(f"  [{i:2d}/{len(selected)}] SKIP (no {args.res} exr): {aid}")
                continue
            size_mb = exr_info.get("size", 0) / (1024 * 1024)
            print(f"  [{i:2d}/{len(selected)}] Downloading {aid} ({size_mb:.1f} MB) [{bucket}] ...", flush=True)
            download_file(exr_info["url"], dst)
            ok += 1
        except Exception as e:
            print(f"  [{i:2d}/{len(selected)}] WARN failed {aid}: {e}", file=sys.stderr)

    print(f"\nDone! {ok} .exr files ready in {args.out}")


if __name__ == "__main__":
    main()
