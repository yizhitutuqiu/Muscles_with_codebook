## REPORT 0511：近期两组消融实验总结（最终结果汇总）

本报告汇总两条线的近期实验：
1) **Pose2EMG（batch_ablation_prior）**：连续/离散先验与语义离散先验的对比（4 组）。指标为 **Average（越小越好）**，并同时对齐官方 `official_cond / official_nocond`。
2) **EMG2Pose（batch_emg2pose_ablation）**：纯 baseline 与 H8 baseline（2 组）。指标为 **Average（越小越好）**，并对齐官方 `official_cond`。

---

## A. Pose2EMG：Prior 系列消融（4 组）

### A.1 实验设置（做了什么）

- **exp_1_pure_continuous**：纯连续分支 baseline（不使用离散 codebook / prior）。
- **exp_2_continuous_prior**：连续分支 + prior（在连续表征上引入先验约束/引导）。
- **exp_3_standard_discrete_prior**：标准离散 codebook + discrete prior（离散化并引入先验）。
- **exp_4_semantic_discrete_prior**：语义离散 codebook + semantic discrete prior（更偏语义的离散化与先验）。

### A.2 Average 汇总（最终结果）

| Experiment | Stage2 Average | Official Cond Average | Official NoCond Average | Δ vs Official Cond |
|---|---:|---:|---:|---:|
| exp_1_pure_continuous | 10.25 | 10.55 | 10.77 | -0.30 |
| exp_2_continuous_prior | 10.15 | 10.55 | 10.77 | -0.40 |
| exp_3_standard_discrete_prior | 10.28 | 10.55 | 10.77 | -0.27 |
| exp_4_semantic_discrete_prior | 10.16 | 10.55 | 10.77 | -0.39 |

**结论（Pose2EMG）**
- 当前 4 组里 **exp_2_continuous_prior（10.15）** 最优，其次 **exp_4_semantic_discrete_prior（10.16）**。
- 四组均优于官方 `official_cond (10.55)` 与 `official_nocond (10.77)`。

### A.3 分动作结果表（Stage2 + Official）

| exercise | exp_1 stage2 | exp_2 stage2 | exp_3 stage2 | exp_4 stage2 | official_cond | official_nocond |
|---|---:|---:|---:|---:|---:|---:|
| ElbowPunch | 11.68 | 11.40 | 11.59 | 11.29 | 11.45 | 12.26 |
| FrontKick | 7.20 | 7.13 | 7.11 | 6.91 | 7.33 | 7.26 |
| FrontPunch | 7.65 | 7.40 | 7.74 | 7.35 | 7.67 | 7.61 |
| HighKick | 9.13 | 9.06 | 9.15 | 9.10 | 9.60 | 9.64 |
| HookPunch | 10.23 | 9.47 | 9.98 | 9.92 | 9.65 | 9.80 |
| JumpingJack | 17.32 | 17.01 | 18.10 | 17.45 | 16.68 | 17.02 |
| KneeKick | 7.39 | 7.36 | 7.26 | 7.21 | 7.71 | 7.77 |
| LegBack | 7.84 | 7.70 | 7.79 | 7.73 | 8.16 | 8.41 |
| LegCross | 6.47 | 6.42 | 6.32 | 6.44 | 7.09 | 6.68 |
| RonddeJambe | 19.47 | 20.06 | 19.42 | 20.07 | 18.93 | 22.65 |
| Running | 7.23 | 7.40 | 7.36 | 7.46 | 7.54 | 7.56 |
| Shuffle | 8.04 | 7.66 | 7.71 | 7.62 | 8.90 | 8.29 |
| SideLunges | 11.20 | 10.95 | 11.21 | 11.23 | 12.77 | 12.28 |
| SlowSkater | 10.06 | 10.23 | 10.12 | 10.06 | 10.91 | 10.71 |
| Squat | 12.91 | 12.96 | 13.27 | 12.63 | 13.86 | 13.54 |
| Average | 10.25 | 10.15 | 10.28 | 10.16 | 10.55 | 10.77 |

---

## B. EMG2Pose：两组消融（2 组）

### B.1 实验设置（做了什么）

- **exp_emg2pose_ablation_pure_continuous**：emg2pose 的纯 baseline（codebook 不参与）。
- **exp_emg2pose_ablation_h8_baseline**：emg2pose 的 H8 baseline（在当前最强结构/配置基础上切换为 emg2pose）。

### B.2 Average 汇总（最终结果）

| Experiment | Stage2 Average | Official Cond Average | Δ vs Official Cond |
|---|---:|---:|---:|
| exp_emg2pose_ablation_pure_continuous | 0.048 | 0.090 | -0.042 |
| exp_emg2pose_ablation_h8_baseline | 0.046 | 0.090 | -0.044 |

**结论（EMG2Pose）**
- 两组均显著优于官方 `official_cond (0.090)`；其中 **H8 baseline（0.046）** 略优于 pure continuous（0.048）。

### B.3 分动作结果表（Stage2 + Official）

| exercise | pure_continuous stage2 | h8_baseline stage2 | official_cond |
|---|---:|---:|---:|
| ElbowPunch | 0.043 | 0.041 | 0.072 |
| FrontKick | 0.044 | 0.041 | 0.073 |
| FrontPunch | 0.035 | 0.034 | 0.068 |
| HighKick | 0.067 | 0.066 | 0.129 |
| HookPunch | 0.039 | 0.036 | 0.076 |
| JumpingJack | 0.043 | 0.042 | 0.089 |
| KneeKick | 0.056 | 0.054 | 0.102 |
| LegBack | 0.050 | 0.049 | 0.090 |
| LegCross | 0.041 | 0.039 | 0.073 |
| RonddeJambe | 0.061 | 0.057 | 0.103 |
| Running | 0.042 | 0.040 | 0.066 |
| Shuffle | 0.050 | 0.049 | 0.074 |
| SideLunges | 0.052 | 0.050 | 0.114 |
| SlowSkater | 0.055 | 0.053 | 0.113 |
| Squat | 0.047 | 0.044 | 0.112 |
| Average | 0.048 | 0.046 | 0.090 |
