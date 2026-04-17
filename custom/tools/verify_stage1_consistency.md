# 为什么 Stage1 单独评测 j3d 激活率 34%，用 Stage2 算就 100%？

## 结论

**两处用的不是同一份 Stage1 权重。**

- **不加 `--stage2_ckpt`**：加载的是**当前磁盘上的** `frame_codebook_frame_for_stage2/best.pt`。
- **加 `--stage2_ckpt`**：用的是 **Stage2 的 checkpoint 里嵌的那份 Stage1**（`payload["model_state"]` 里所有 `stage1.*` 的权重）。

Stage2 加载流程是：

1. 用 `stage1_path`（如 best.pt）**先**建一个 Stage1 并 `load_state_dict(best.pt)`；
2. 再 `model.load_state_dict(payload["model_state"], strict=True)` 把**整网**权重加载进来。

整网里包含 `stage1.joints3d.*`、`stage1.vq.*` 等，所以会**覆盖**掉上一步从 best.pt 读进来的权重。  
因此最终跑推理时，Stage2 里的 Stage1 = **当时存 Stage2 时的那份 Stage1 快照**，而不是「当前 best.pt」。

若在存完 Stage2 之后又：

- 重新训练了 Stage1 并覆盖了 `best.pt`，或  
- 换过/覆盖过 Stage1 的 checkpoint，

那么：

- **当前 best.pt** → 单独评测 → 34% 激活率；
- **Stage2 里嵌的那份 Stage1** → 用 `--stage2_ckpt` 评测 → 100% 激活率。

所以从实现上两者本来就可以不一致；理论上「冻结 Stage1 训练 Stage2」只保证**训练时**用的是当时的那份 Stage1，存进 Stage2 的也是那份，并不保证和**现在的** best.pt 相同。

## 如何验证

同目录下脚本 `verify_stage1_in_stage2_ckpt.py` 可对比两份权重是否一致：

```bash
cd /path/to/musclesinaction
python custom/tools/verify_stage1_in_stage2_ckpt.py \
  --stage1_ckpt custom/checkpoints/frame_codebook_frame_for_stage2/best.pt \
  --stage2_ckpt custom/stage2/checkpoints/stage2_pose2emg/best.pt
```

- 若输出 **「Stage1 权重一致」**：说明当前 best.pt 与 Stage2 里嵌的 Stage1 相同，此时 34% vs 100% 就可能是别的原因（例如输入/数据路径差异）。
- 若输出 **「Stage1 权重不一致」**：说明存 Stage2 之后 best.pt 被覆盖或换过，两处本来就不是同一份 Stage1，出现 34% vs 100% 正常。
