#!/usr/bin/env python3
"""
诊断 checkpoint 中归一化层的 running stats，用于排查 eval 时 Loss 爆炸。

用法:
  python custom/tools/inspect_codebook_checkpoint_stats.py --checkpoint path/to/best.pt

若看到 mean/var 或 running_mean/running_var 全为 0、全为 1、或 min/max 异常，
则说明 OnlineStandardizer 或 TCN 的 BatchNorm 统计量损坏，eval 时会出问题。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect normalization buffers in frame codebook checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    args = parser.parse_args()

    path = Path(args.checkpoint).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state", payload)
    if not isinstance(state, dict):
        print("Checkpoint has no model_state dict.")
        return

    # 1) OnlineStandardizer: mean, var, initialized (名字是 mean/var 不是 running_*)
    # 2) BatchNorm1d: running_mean, running_var
    keywords = ("mean", "var", "running_mean", "running_var", "initialized")
    found = []
    for key in sorted(state.keys()):
        for kw in keywords:
            if kw in key:
                found.append((key, kw))
                break

    if not found:
        print("No mean/var / running_mean/running_var / initialized keys in checkpoint.")
        return

    print("=" * 80)
    print("Normalization-related buffers in checkpoint (min/max/sum)")
    print("=" * 80)
    for key, _ in found:
        t = state[key]
        if not isinstance(t, torch.Tensor):
            print(f"  {key}: (not a tensor)")
            continue
        t = t.float()
        min_v = t.min().item()
        max_v = t.max().item()
        sum_v = t.sum().item()
        numel = t.numel()
        mean_v = sum_v / numel if numel else 0.0
        # 若为 bool (initialized)
        if t.dtype == torch.bool or key == "initialized":
            print(f"  {key}: value={t.item() if t.numel() == 1 else (t.any().item(), t.all().item())}")
        else:
            print(f"  {key}: min={min_v:.6g}, max={max_v:.6g}, mean={mean_v:.6g}, numel={numel}")
        if "var" in key or "running_var" in key:
            if max_v < 1e-6:
                print(f"       ⚠️  variance 接近 0，eval 时除以 sqrt(var) 会导致特征爆炸")
            elif min_v < 1e-6:
                print(f"       ⚠️  存在接近 0 的分量，可能被 padding/异常值污染")

    print("=" * 80)


if __name__ == "__main__":
    main()
