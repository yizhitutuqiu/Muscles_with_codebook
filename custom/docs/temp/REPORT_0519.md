## 0519 — Lightweight Stage1 + Official Lightweight Transformer Backbone（Stage2）

目标：在保持 Stage2 时序主干为 **MIA 官方轻量 TransformerEnc（架构/超参/参数量严格一致）** 的前提下，将 Stage1 codebook/encoder 替换为轻量版 `exp_shared_head_h512_l3`（hidden_dim=512，TCN layers=3），验证：

- Stage1 轻量化后，离散先验 + 融合策略是否仍有效
- Stage1 轻量化是否会影响（甚至提升）泛化指标
- 同时量化推理侧的参数规模收益

本次对比使用两组 Stage2 实验结果：
- 0518（Stage1=exp_shared_head）：[/REPORT_0518.md](file:///data/litengmo/HSMR/mia_custom/custom/docs/temp/REPORT_0518.md)
- 0519（Stage1=exp_shared_head_h512_l3）：本次新跑结果
同时补充两个“无离散先验”的超大模型对照（纯 DSTFormer + MoE）：
- pose2emg：`custom/ablations/batch_ablation_prior/exp_1_pure_continuous`
- emg2pose：`custom/stage2/checkpoints/batch_emg2pose_ablation/exp_emg2pose_ablation_pure_continuous`

### 参数量统计口径（按“实际推理用到的 Stage1”）

Stage1 在 Stage2 推理时只用于 **生成离散 token**：
- pose2emg：只会用到 `Stage1.joints3d.encoder`（以及 VQ 的 buffer 状态；VQ EMA 在 parameters 口径下为 0）
- emg2pose：只会用到 `Stage1.emg.encoder`

因此下表中的 “Stage1 params” 只统计对应 encoder 的参数量（不把另一个模态的 encoder/decoder 计入）。

---

## 总结表（Average + Params）

### pose2emg（RMSE，越低越好）

| | 0518：Stage1=exp_shared_head | 0519：Stage1=exp_shared_head_h512_l3 | PureContinuous（DSTFormer+MoE，无离散先验） | Official(cond) | Official(nocond) |
|---|---:|---:|---:|---:|---:|
| Average RMSE | 10.41 | 10.37 | 10.25 | 10.55 | 10.77 |
| Params（推理口径） | 19.32M（Stage1=12.94M + Stage2=6.39M） | 8.92M（Stage1=2.53M + Stage2=6.39M） | 109.36M（Stage2 only） | 5.42M | 5.42M |

### emg2pose（MPJPE，越低越好）

| | 0518：Stage1=exp_shared_head | 0519：Stage1=exp_shared_head_h512_l3 | PureContinuous（DSTFormer+MoE，无离散先验） | Official(cond) | Official(nocond) |
|---|---:|---:|---:|---:|---:|
| Average MPJPE | 0.054 | 0.055 | 0.048 | 0.090 | 0.090 |
| Params（推理口径） | 19.18M（Stage1=12.87M + Stage2=6.32M） | 8.81M（Stage1=2.50M + Stage2=6.32M） | 108.25M（Stage2 only） | 5.36M | 5.36M |

说明：
- emg2pose 的 Official(nocond) 本质是“同一 cond 权重 + dataloader cond=False”的评测口径；该模型对 cond 的敏感度非常弱，3 位小数下 cond/nocond 会显示为相同数值。

---

### 观察与结论

- **与超大纯连续模型对比**：纯 DSTFormer+MoE（无离散先验）确实能把指标进一步压低（pose2emg=10.25、emg2pose=0.048），但代价是 Stage2 参数量约 **108–109M**，远高于我们“官方轻量 Transformer 主干 + 离散先验融合”的 **6.3–6.4M** Stage2。
- **pose2emg**：Stage1 轻量化后，Average RMSE 从 10.41 → **10.37**（小幅提升），同时推理参数量从 19.32M → **8.92M**（约 54% 减少）。
- **emg2pose**：Stage1 轻量化后，Average MPJPE 从 0.054 → 0.055（几乎不变的量级），但推理参数量从 19.18M → **8.81M**（约 54% 减少）。
- 结论：在 **Stage2 主干固定为官方轻量 Transformer** 的条件下，Stage1 显著轻量化依然能保持（甚至略提升）泛化指标，且带来明显的推理参数规模收益；相比超大纯连续 MoE 模型，我们在参数效率上更有优势。
