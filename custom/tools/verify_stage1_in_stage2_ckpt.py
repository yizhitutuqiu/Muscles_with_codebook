#!/usr/bin/env python3
"""
对比「Stage2 里嵌的 Stage1」与「单独 Stage1 checkpoint」的权重是否一致。
用法:
  python verify_stage1_in_stage2_ckpt.py \\
    --stage1_ckpt custom/checkpoints/frame_codebook_frame_for_stage2/best.pt \\
    --stage2_ckpt custom/stage2/checkpoints/stage2_pose2emg/best.pt
若一致会打印 "Stage1 权重一致"；否则会列出差异（key 或 value 不同）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage1 inside Stage2 ckpt vs standalone Stage1 ckpt.")
    parser.add_argument("--stage1_ckpt", type=str, required=True, help="Standalone Stage1 checkpoint (e.g. frame_codebook.../best.pt)")
    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Stage2 checkpoint that embeds Stage1")
    args = parser.parse_args()

    p1 = Path(args.stage1_ckpt).expanduser().resolve()
    p2 = Path(args.stage2_ckpt).expanduser().resolve()
    if not p1.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {p1}")
    if not p2.exists():
        raise FileNotFoundError(f"Stage2 checkpoint not found: {p2}")

    s1_payload = torch.load(p1, map_location="cpu")
    s2_payload = torch.load(p2, map_location="cpu")

    state1 = s1_payload.get("model_state") or s1_payload
    if not isinstance(state1, dict):
        raise RuntimeError(f"Stage1 checkpoint has no model_state dict: {type(state1)}")

    state2_full = s2_payload.get("model_state")
    if not isinstance(state2_full, dict):
        raise RuntimeError(f"Stage2 checkpoint has no model_state: {type(state2_full)}")
    state2_stage1 = {k.removeprefix("stage1."): v for k, v in state2_full.items() if k.startswith("stage1.")}

    if not state2_stage1:
        print("Stage2 checkpoint 中未找到 stage1.* 的 key，无法对比。")
        return

    keys1 = set(state1.keys())
    keys2 = set(state2_stage1.keys())
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    if only_in_1 or only_in_2:
        print("Key 不一致:")
        if only_in_1:
            print("  仅在 Stage1 中:", sorted(only_in_1)[:20], "..." if len(only_in_1) > 20 else "")
        if only_in_2:
            print("  仅在 Stage2.stage1 中:", sorted(only_in_2)[:20], "..." if len(only_in_2) > 20 else "")
        return

    diffs = []
    for k in sorted(keys1):
        v1 = state1[k]
        v2 = state2_stage1[k]
        if v1.shape != v2.shape:
            diffs.append((k, "shape", str(v1.shape), str(v2.shape)))
        elif not torch.allclose(v1.float(), v2.float(), rtol=0, atol=0):
            diff = (v1.float() - v2.float()).abs().max().item()
            diffs.append((k, "value", f"max_diff={diff}", ""))

    if not diffs:
        print("Stage1 权重一致（Stage2 里嵌的 Stage1 与当前 standalone best.pt 相同）。")
        return
    print(f"Stage1 权重不一致，共 {len(diffs)} 个 key 不同:")
    for item in diffs[:30]:
        print(" ", item)
    if len(diffs) > 30:
        print("  ...")


if __name__ == "__main__":
    main()
