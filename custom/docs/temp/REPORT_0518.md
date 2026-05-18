## 0518 — Official Lightweight Transformer Backbone 验证（Stage2）

目标：证明我们的 **离散先验（Stage1 codebook）+ 连续分支 + 融合策略（symmetric DCSA）** 的收益，不依赖于 DSTFormer / MoE 等复杂时序主干。为此我们把 Stage2 的时序主干 **完全替换为 musclesinaction 官方轻量 TransformerEnc**（架构与超参一致，且参数量严格一致），然后对 pose2emg / emg2pose 两个任务分别评测并与官方模型对比。

### 平均结果总览（只看 Average）

#### pose2emg（RMSE，越低越好）

| stage2 | official_cond | official_nocond |
|---:|---:|---:|
| 10.41 | 10.55 | 10.77 |

#### emg2pose（MPJPE，越低越好）

| stage2 | official_cond |
|---:|---:|
| 0.054 | 0.090 |

### 实验设定（关键点）

- Stage1 checkpoint：`custom/checkpoints/new_clip5_codebook_with_exerciseloss/exp_shared_head/best.pt`
- Stage2：保留离散/连续双分支与 symmetric DCSA 融合；**temporal backbone 替换为官方 TransformerEnc**；cond=False（不使用条件输入）
- 官方 backbone 参数量对齐（严格一致）：
  - pose2emg 官方 TransformerEnc：5,424,712 params
  - emg2pose 官方 TransformerEnc：5,357,361 params
- 结果来源：
  - pose2emg：`custom/stage2/checkpoints/stage2_clip5_official_transformer_pose2emg/eval_results/summary_processed.csv`
  - emg2pose：`custom/stage2/checkpoints/stage2_clip5_official_transformer_emg2pose/eval_results/summary_processed.csv`

### 核心结论（更有说服力的证据）

- **pose2emg（RMSE）**：在主干完全换成官方轻量 Transformer 后，我们的 Stage2 仍能超过官方模型
  - Average：Stage2 = 10.41，Official(cond) = 10.55（绝对提升 0.14），Official(nocond) = 10.77（绝对提升 0.36）
- **emg2pose（MPJPE）**：同样在官方轻量 Transformer 主干下，融合策略收益更明显
  - Average：Stage2 = 0.054，Official(cond) = 0.090（绝对提升 0.036）

换句话说：**即使把时序主干替换为“与官方完全一致（架构/超参/参数量）”的轻量 Transformer，我们的离散先验融合策略依然带来稳定可复现的收益**；收益来源于“先验+融合”，而不是“更重的时序网络”。

---

## 结果表格

### pose2emg（RMSE，越低越好）

| exercise | stage2 | official_cond | official_nocond |
|---|---:|---:|---:|
| ElbowPunch | 11.78 | 11.45 | 12.26 |
| FrontKick | 7.26 | 7.33 | 7.26 |
| FrontPunch | 7.47 | 7.67 | 7.61 |
| HighKick | 9.15 | 9.60 | 9.64 |
| HookPunch | 9.96 | 9.65 | 9.80 |
| JumpingJack | 16.34 | 16.68 | 17.02 |
| KneeKick | 7.40 | 7.71 | 7.77 |
| LegBack | 8.06 | 8.16 | 8.41 |
| LegCross | 6.79 | 7.09 | 6.68 |
| RonddeJambe | 21.54 | 18.93 | 22.65 |
| Running | 7.42 | 7.54 | 7.56 |
| Shuffle | 8.42 | 8.90 | 8.29 |
| SideLunges | 11.33 | 12.77 | 12.28 |
| SlowSkater | 10.20 | 10.91 | 10.71 |
| Squat | 13.05 | 13.86 | 13.54 |
| Average | 10.41 | 10.55 | 10.77 |

### emg2pose（MPJPE，越低越好）

| exercise | stage2 | official_cond |
|---|---:|---:|
| ElbowPunch | 0.048 | 0.072 |
| FrontKick | 0.048 | 0.073 |
| FrontPunch | 0.044 | 0.068 |
| HighKick | 0.074 | 0.129 |
| HookPunch | 0.046 | 0.076 |
| JumpingJack | 0.053 | 0.089 |
| KneeKick | 0.059 | 0.102 |
| LegBack | 0.057 | 0.090 |
| LegCross | 0.047 | 0.073 |
| RonddeJambe | 0.064 | 0.103 |
| Running | 0.045 | 0.066 |
| Shuffle | 0.055 | 0.074 |
| SideLunges | 0.059 | 0.114 |
| SlowSkater | 0.061 | 0.113 |
| Squat | 0.053 | 0.112 |
| Average | 0.054 | 0.090 |
