#!/usr/bin/env python3
"""
Train a D4-invariant 9->4->4->1 (56 FMA) HDR-native neural sharpening model in PyTorch
and export drop-in HLSL code matching RunModelA_Fast9() in Shaders/Neural_Sharpen_HDR.fx.

Uses a 4-part HDR perceptual loss:
  1. Charbonnier PQ accuracy loss
  2. Asymmetric OLED halo penalty (2.5x stricter on bright overshoot than dark undershoot)
  3. Linear-nit energy conservation (prevents artificial foliage brightening)
  4. Explicit zero-DC pure-black/flat and 1-px specular firefly suppression

Example:
    python train_hdr_cnn.py --data dataset/patches_stationary.npz --epochs 60 --out exported_fast9_hlsl.txt
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESIDUAL_SCALE = 0.22  # Matches `clamp(rawResidual, -0.5, 0.5) * 0.22` in Neural_Sharpen_HDR.fx


def fast_pq_to_linear_torch(pq: torch.Tensor) -> torch.Tensor:
    """Differentiable 0-pow ST.2084 PQ -> Linear [0, 1] rational fit (matches shader math)."""
    v = torch.clamp(pq, 0.0, 1.0)
    num = v * (1.811813 + v * (8.965278 + v * 1.185166))
    den = 1.0 + v * (27.571310 - v * 16.625550)
    u = F.relu(num / den)
    u2 = u * u
    return u2 * u2


def extract_d4_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the exact 9 D4-invariant features used by Profile 1 (RunModelA_Fast9) in Neural_Sharpen_HDR.fx.
    Input x: (N, 9) ordered [p0..p8].
    Returns (feats (N, 9), max3x3 (N,), min3x3 (N,)).
    """
    p0, p1, p2 = x[:, 0], x[:, 1], x[:, 2]
    p3, p4, p5 = x[:, 3], x[:, 4], x[:, 5]
    p6, p7, p8 = x[:, 6], x[:, 7], x[:, 8]

    max_cross = torch.maximum(torch.maximum(p1, p7), torch.maximum(p3, p5))
    min_cross = torch.minimum(torch.minimum(p1, p7), torch.minimum(p3, p5))
    max_diag  = torch.maximum(torch.maximum(p0, p2), torch.maximum(p6, p8))
    min_diag  = torch.minimum(torch.minimum(p0, p2), torch.minimum(p6, p8))

    max3 = torch.maximum(torch.maximum(max_cross, max_diag), p4)
    min3 = torch.minimum(torch.minimum(min_cross, min_diag), p4)

    avg_cross_diff = 0.25 * ((p1 + p7) + (p3 + p5)) - p4
    avg_diag_diff  = 0.25 * ((p0 + p8) + (p2 + p6)) - p4
    aniso_hv       = 0.5  * torch.abs((p1 + p7) - (p3 + p5))
    aniso_diag     = 0.5  * torch.abs((p0 + p8) - (p2 + p6))

    feats = torch.stack([
        p4,
        avg_cross_diff, avg_diag_diff,
        max_cross - p4, min_cross - p4,
        max_diag  - p4, min_diag  - p4,
        aniso_hv,       aniso_diag,
    ], dim=1)
    return feats, max3, min3


class D4Fast9Net(nn.Module):
    """Exact 9 -> 4 -> 4 -> 1 PReLU architecture (56 FMAs, 1 float4 register/layer)."""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(9, 4)
        self.a1  = nn.Parameter(torch.full((4,), -0.25))
        self.fc2 = nn.Linear(4, 4)
        self.a2  = nn.Parameter(torch.full((4,), -0.25))
        self.fc3 = nn.Linear(4, 1)

        nn.init.kaiming_uniform_(self.fc1.weight, a=0.25)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.25)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc3.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h1 = self.fc1(f)
        h1 = torch.where(h1 >= 0.0, h1, h1 * self.a1)
        h2 = self.fc2(h1)
        h2 = torch.where(h2 >= 0.0, h2, h2 * self.a2)
        return self.fc3(h2).squeeze(-1)


def hdr_perceptual_loss(
    raw_res: torch.Tensor,
    p4: torch.Tensor,
    y_gt: torch.Tensor,
    max3: torch.Tensor,
    min3: torch.Tensor,
    kind: torch.Tensor,
    halo_weight: float,
    energy_weight: float,
) -> torch.Tensor:
    # Smooth clamp during training matching clamp(rawResidual, -0.5, 0.5) * 0.22
    delta = torch.clamp(raw_res, -0.5, 0.5) * RESIDUAL_SCALE
    pred  = torch.clamp(p4 + delta, 0.0, 1.0)
    rng3  = max3 - min3

    is_real    = (kind == 0).float()
    is_flat    = (kind == 1).float()
    is_firefly = (kind == 2).float()

    # 1. Charbonnier PQ accuracy loss on real HDR patches
    diff = pred - y_gt
    charb = torch.sqrt(diff * diff + 1e-6)
    loss_acc = (charb * is_real).sum() / (is_real.sum() + 1e-6)

    # 2. Asymmetric OLED Halo Penalty (2.5x stricter on bright overshoot than dark undershoot,
    #    scaled up on higher-contrast edges where silhouette halos are most visible!)
    edge_w = 0.4 + 2.5 * torch.clamp((rng3 - 0.06) / 0.18, 0.0, 1.0)
    # Allow overshoot only up to whatever the clean ground truth itself exceeded max3/min3 by
    gt_pos_ov = F.relu(y_gt - max3)
    gt_neg_ov = F.relu(min3 - y_gt)
    excess_pos = F.relu(F.relu(pred - max3) - gt_pos_ov)
    excess_neg = F.relu(F.relu(min3 - pred) - gt_neg_ov)
    halo_pen = (2.5 * excess_pos + 1.0 * excess_neg) * edge_w
    loss_halo = (halo_pen * is_real).sum() / (is_real.sum() + 1e-6)

    # 3. Linear-Nit Energy Conservation (prevents distant sunlit foliage from brightening artificially)
    lin_pred = fast_pq_to_linear_torch(pred)
    lin_gt   = fast_pq_to_linear_torch(y_gt)
    mean_drift = torch.abs(((lin_pred - lin_gt) * is_real).sum() / (is_real.sum() + 1e-6))

    # 4. Zero-DC pure-black/flat and 1-px specular firefly suppression
    loss_flat = (torch.abs(delta) * is_flat).sum() / (is_flat.sum() + 1e-6)
    loss_ff   = (torch.abs(delta) * is_firefly).sum() / (is_firefly.sum() + 1e-6)

    return loss_acc + halo_weight * loss_halo + energy_weight * mean_drift + 4.0 * loss_flat + 2.0 * loss_ff


def fmt_f4(vec: np.ndarray) -> str:
    return f"NS_F4({vec[0]:.9f}, {vec[1]:.9f}, {vec[2]:.9f}, {vec[3]:.9f})"


def export_hlsl(model: D4Fast9Net, fn_name: str) -> str:
    w1 = model.fc1.weight.detach().cpu().numpy()  # (4, 9)
    b1 = model.fc1.bias.detach().cpu().numpy()    # (4,)
    a1 = model.a1.detach().cpu().numpy()          # (4,)
    w2 = model.fc2.weight.detach().cpu().numpy()  # (4, 4)
    b2 = model.fc2.bias.detach().cpu().numpy()    # (4,)
    a2 = model.a2.detach().cpu().numpy()          # (4,)
    w3 = model.fc3.weight.detach().cpu().numpy()[0]  # (4,)
    b3 = float(model.fc3.bias.detach().cpu().numpy()[0])

    lines = [
        f"NS_F1 {fn_name}(",
        "    NS_F1 f0, NS_F1 f1, NS_F1 f2, NS_F1 f3, NS_F1 f4,",
        "    NS_F1 f5, NS_F1 f6, NS_F1 f7, NS_F1 f8)",
        "{",
        "    // Layer 1: 9 D4-invariant features -> 4 (9 float4 FMAs, 1 float4 register)",
        f"    NS_F4 L1 = {fmt_f4(b1)};",
    ]
    for i in range(9):
        lines.append(f"    L1 += f{i} * {fmt_f4(w1[:, i])};")
    lines += [
        f"    L1 = PReLU4(L1, {fmt_f4(a1)});",
        "",
        "    // Layer 2: 4 -> 4 (4 float4 FMAs, 1 float4 register)",
        f"    NS_F4 L2 = {fmt_f4(b2)};",
    ]
    comps = ("x", "y", "z", "w")
    for i, c in enumerate(comps):
        lines.append(f"    L2 += L1.{c} * {fmt_f4(w2[:, i])};")
    lines += [
        f"    L2 = PReLU4(L2, {fmt_f4(a2)});",
        "",
        "    // Layer 3: 4 -> 1 (1 float4 mul + sum)",
        f"    NS_F4 outAcc = L2 * {fmt_f4(w3)};",
        f"    return (NS_F1){b3:.9f} + (outAcc.x + outAcc.y) + (outAcc.z + outAcc.w);",
        "}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HDR-native D4 Fast9 CNN and export HLSL weights.")
    parser.add_argument("--data", default="dataset/patches_stationary.npz", help="Input .npz from prepare_hdr_dataset.py")
    parser.add_argument("--epochs", type=int, default=60, help="AdamW epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size (default: 16384)")
    parser.add_argument("--lr", type=float, default=3e-3, help="Initial learning rate (default: 3e-3)")
    parser.add_argument("--halo-weight", type=float, default=0.85, help="Asymmetric OLED halo penalty weight (default: 0.85)")
    parser.add_argument("--energy-weight", type=float, default=1.50, help="Linear-nit mean drift penalty weight (default: 1.50)")
    parser.add_argument("--fn-name", default="RunModelA_Fast9", help="Exported HLSL function name")
    parser.add_argument("--out", default="exported_fast9_hlsl.txt", help="Output HLSL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    data = np.load(args.data)
    x_np = data["x"].astype(np.float32)
    y_np = data["y"].astype(np.float32)
    k_np = data["kind"].astype(np.uint8)
    n_total = x_np.shape[0]
    n_val = max(20000, n_total // 10)
    n_tr  = n_total - n_val
    print(f"Loaded {n_total:,} patches ({n_tr:,} train / {n_val:,} val) from {args.data}")

    # Precompute exact D4 features on GPU once (tiny memory footprint, blazing fast epochs!)
    with torch.no_grad():
        x_tr_t = torch.from_numpy(x_np[:n_tr]).to(device)
        f_tr, max_tr, min_tr = extract_d4_features(x_tr_t)
        p4_tr = x_tr_t[:, 4].contiguous()
        y_tr  = torch.from_numpy(y_np[:n_tr]).to(device)
        k_tr  = torch.from_numpy(k_np[:n_tr]).to(device)
        del x_tr_t

        x_va_t = torch.from_numpy(x_np[n_tr:]).to(device)
        f_va, max_va, min_va = extract_d4_features(x_va_t)
        p4_va = x_va_t[:, 4].contiguous()
        y_va  = torch.from_numpy(y_np[n_tr:]).to(device)
        k_va  = torch.from_numpy(k_np[n_tr:]).to(device)
        del x_va_t

    model = D4Fast9Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

    t0 = time.time()
    best_val = 1e9
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            raw = model(f_tr[idx])
            loss = hdr_perceptual_loss(
                raw, p4_tr[idx], y_tr[idx], max_tr[idx], min_tr[idx], k_tr[idx],
                args.halo_weight, args.energy_weight,
            )
            loss.backward()
            opt.step()
        sched.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                va_raw = model(f_va)
                va_loss = float(hdr_perceptual_loss(
                    va_raw, p4_va, y_va, max_va, min_va, k_va,
                    args.halo_weight, args.energy_weight,
                ).item())
                if va_loss < best_val:
                    best_val = va_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                # Diagnostic metrics on real val patches
                real_m = (k_va == 0)
                va_delta = torch.clamp(va_raw, -0.5, 0.5) * RESIDUAL_SCALE
                va_pred = torch.clamp(p4_va + va_delta, 0.0, 1.0)
                in_mae   = float(torch.abs(p4_va[real_m] - y_va[real_m]).mean().item())
                pred_mae = float(torch.abs(va_pred[real_m] - y_va[real_m]).mean().item())
                pos_ov   = float(F.relu(va_pred[real_m] - max_va[real_m]).mean().item())
                blk_err  = float(torch.abs(va_delta[k_va == 1]).max().item()) if (k_va == 1).any() else 0.0
                ff_err   = float(torch.abs(va_delta[k_va == 2]).mean().item()) if (k_va == 2).any() else 0.0
            print(
                f"Epoch {epoch:3d}/{args.epochs} ({time.time()-t0:5.1f}s) | "
                f"val_loss={va_loss:.5f} | MAE: {in_mae:.5f} -> {pred_mae:.5f} "
                f"({(1.0 - pred_mae/max(in_mae,1e-9))*100:+.1f}%) | "
                f"avg_bright_ov={pos_ov:.5f} | flat_max={blk_err:.5f} | ff_mae={ff_err:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Export drop-in HLSL function
    hlsl_code = export_hlsl(model, args.fn_name)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(hlsl_code)
    print(f"\nExported drop-in HLSL ({sum(p.numel() for p in model.parameters())} params, 56 FMAs) -> {args.out}")
    print("Paste the generated function directly over RunModelA_Fast9() in Shaders/Neural_Sharpen_HDR.fx!")


if __name__ == "__main__":
    main()
