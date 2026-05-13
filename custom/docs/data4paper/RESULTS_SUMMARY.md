# data4paper：可用于论文展示的核心实验结果（summary_processed.csv）

本目录用于收拢当前阶段“可写进论文”的核心结果文件（以每个实验的 `summary_processed.csv` 为主），并提供一个只看均值（Average）的总表，以及每个实验的来源、目的与关键配置说明。

评测协议统一说明：
- **protocol**：`id_exercises`（val split）
- **step**：30
- **joints3d_root_center**：true
- **指标**：`Average`（越小越好）
  - pose2emg：与官方对齐的 RMSE（与我们历史表格一致的量级，约 10 左右）
  - emg2pose：与官方对齐的 MPJPE（量级约 0.0x）

---

## 1) Pose2EMG：VQ-VAE prior 消融（batch_ablation_prior）

### 1.0 主结果（论文主表建议）

| Experiment | Stage2 Avg | Official Cond Avg | Official NoCond Avg |
|---|---:|---:|---:|
| exp_h8_baseline_8exp (best) | 10.08 | 10.55 | 10.77 |

对应文件：
- pose2emg_main_h8_baseline_8exp_best_summary_processed.csv
- pose2emg_main_h8_baseline_8exp_best_train_config.yaml

### 1.1 均值总表（Average）

| ID | Experiment | Stage2 Avg | Official Cond Avg | Official NoCond Avg | 说明 |
|---|---|---:|---:|---:|---|
| P1 | exp_1_pure_continuous | 10.25 | 10.55 | 10.77 | 纯连续 baseline，不使用离散先验（fusion=none） |
| P2 | exp_2_continuous_prior_encoder_only | 10.82 | 10.55 | 10.77 | **禁用 VQ 量化**，将 encoder 输出 `z_e` 当作 `z_disc` 参与融合（symmetric_dcsa） |
| P3 | exp_3_standard_discrete_prior | 10.21 | 10.55 | 10.77 | 标准离散 codebook prior（clip_5_code_256）+ symmetric_dcsa |
| P4 | exp_4_semantic_discrete_prior | 10.16 | 10.55 | 10.77 | 语义 codebook prior（带 exercise 辅助分类训练的 codebook）+ symmetric_dcsa |

### 1.2 文件（已复制到本目录）

- P1：pose2emg_exp1_pure_continuous_summary_processed.csv、pose2emg_exp1_pure_continuous_train_config.yaml
- P2：pose2emg_exp2_continuous_prior_encoder_only_summary_processed.csv、pose2emg_exp2_continuous_prior_encoder_only_train_config.yaml
- P3：pose2emg_exp3_standard_discrete_prior_summary_processed.csv、pose2emg_exp3_standard_discrete_prior_train_config.yaml
- P4：pose2emg_exp4_semantic_discrete_prior_summary_processed.csv、pose2emg_exp4_semantic_discrete_prior_train_config.yaml

### 1.3 实验来源、目的与关键配置

- **主结果 exp_h8_baseline_8exp（best）**
  - 来源目录：`custom/stage2/checkpoints/batch_stage2_h8_lightweight_0511/exp_h8_baseline_8exp/`
  - 目的：作为当前阶段 Pose2EMG 的主结果（与官方对齐的最优 RMSE），用于论文主表展示。
  - 关键配置（摘要）：`stage1.checkpoint=clip_5_code_256/best.pt`；`cont_encoder_type=joint_25`；`fusion_type=symmetric_dcsa`；`temporal_type=dstformer_v5_guided_moe(num_experts=8, guide_mode=none)`；`batch_size=16`。

- **P1 exp_1_pure_continuous**
  - 来源目录：`custom/ablations/batch_ablation_prior/exp_1_pure_continuous/`
  - 目的：验证“完全不使用离散 codebook / prior”时，仅靠连续分支 + temporal backbone 的表现下界。
  - 关键配置（摘要）：`methods.stage1.encoder_decoder_only=true`；`fusion_type=none`；`temporal_type=dstformer_v5_guided_moe(num_experts=8, guide_mode=none)`；`stage1.checkpoint=clip_5_code_256/best.pt`。

- **P2 exp_2_continuous_prior（本次为 encoder-only 真正生效版）**
  - 来源目录：`custom/ablations/batch_ablation_prior/exp_2_continuous_prior/`
  - 目的：仅去掉“离散量化”（VQ），但保留 Stage1 encoder 与融合策略，检验提升是否来自 encoder 本身而非离散 codebook。
  - 关键配置（摘要）：`methods.stage1.encoder_decoder_only=true`；`fusion_type=symmetric_dcsa`；`temporal_type=dstformer_v5_guided_moe(num_experts=8, guide_mode=none)`；`stage1.checkpoint=clip_5_code_256/best.pt`。
  - 关键实现说明：当 `encoder_decoder_only=true` 时，Stage2 的离散分支 token 直接返回 `z_e` 作为 `z_disc`（不再调用 `vq(z_e)`）。

- **P3 exp_3_standard_discrete_prior**
  - 来源目录：`custom/ablations/batch_ablation_prior/exp_3_standard_discrete_prior/`
  - 目的：标准离散 codebook prior 的基线（对齐 H8 baseline 的“使用离散 token”路径）。
  - 关键配置（摘要）：`methods.stage1.checkpoint=custom/checkpoints/batch_codebook/clip_5_code_256/best.pt`；`fusion_type=symmetric_dcsa`；`temporal_type=dstformer_v5_guided_moe(num_experts=8, guide_mode=none)`。

- **P4 exp_4_semantic_discrete_prior**
  - 来源目录：`custom/ablations/batch_ablation_prior/exp_4_semantic_discrete_prior/`
  - 目的：语义 codebook prior（带 exercise 辅助分类损失训练的 codebook）作为更强的离散先验上界参考。
  - 关键配置（摘要）：`methods.stage1.checkpoint=custom/checkpoints/new_clip5_codebook_with_exerciseloss/exp_shared_head/best.pt`；`fusion_type=symmetric_dcsa`；`temporal_type=dstformer_v5_guided_moe(num_experts=8, guide_mode=none)`。

---

## 2) EMG2Pose：反向任务消融（batch_emg2pose_ablation）

### 2.1 均值总表（Average）

| ID | Experiment | Stage2 Avg | Official Cond Avg | 说明 |
|---|---|---:|---:|---|
| E1 | exp_emg2pose_ablation_pure_continuous | 0.048 | 0.090 | emg2pose 纯 baseline（codebook 不参与） |
| E2 | exp_emg2pose_ablation_h8_baseline | 0.046 | 0.090 | emg2pose 的 H8 baseline（保留有效结构/技巧，切换为 emg2pose） |

### 2.2 文件（已复制到本目录）

- E1：emg2pose_pure_continuous_summary_processed.csv、emg2pose_pure_continuous_train_config.yaml
- E2：emg2pose_h8_baseline_summary_processed.csv、emg2pose_h8_baseline_train_config.yaml

### 2.3 实验来源、目的与关键配置

- **E1 exp_emg2pose_ablation_pure_continuous**
  - 来源目录：`custom/stage2/checkpoints/batch_emg2pose_ablation/exp_emg2pose_ablation_pure_continuous/`
  - 目的：emg2pose 任务下的纯 baseline，对齐官方 metric（MPJPE）与量级。
  - 关键配置（摘要）：`model.task=emg2pose`；`stage1.checkpoint=clip_5_code_256/best.pt`；其余结构与 pose2emg 的 Stage2 框架一致。

- **E2 exp_emg2pose_ablation_h8_baseline**
  - 来源目录：`custom/stage2/checkpoints/batch_emg2pose_ablation/exp_emg2pose_ablation_h8_baseline/`
  - 目的：把当前最强的 H8 结构/技巧迁移到 emg2pose，观察反向回归的收益。
  - 关键配置（摘要）：`model.task=emg2pose`；结构与 H8 baseline 对齐（同一 Stage2 框架，保持关键模块）。
