## 0529 — ID / OOD Final Report（Our vs Official）

本报告汇总并对比：

- Our 模型（Stage2=官方轻量 TransformerEnc 主干 + 离散先验融合）
- Official 模型（musclesinaction 官方训练/评测口径）

任务与指标：

- pose2emg：RMSE（越低越好）
- emg2pose：MPJPE（越低越好）

口径说明：

- Our Stage2 全部为 use_cond=false；因此 Official 侧统一取 nocond 指标对比。
- ID：只统计按运动（exercise）的 val split。
- OOD：分别统计
  - OOD-exercise：按运动类型划分（每个运动独立训练一个模型）。
  - OOD-person：按 Subject 划分的 leave-one-subject-out（每个 Subject 独立训练一个模型，训练集剔除该 Subject）。

---

### 1) ID（按运动）

|  | Our pose2emg | Official pose2emg | Our emg2pose | Official emg2pose |
|---|---:|---:|---:|---:|
| ElbowPunch | 11.62 | 12.26 | 0.040 | 0.072 |
| FrontKick | 7.14 | 7.26 | 0.042 | 0.073 |
| FrontPunch | 7.44 | 7.61 | 0.035 | 0.068 |
| HighKick | 9.09 | 9.64 | 0.067 | 0.129 |
| HookPunch | 9.53 | 9.80 | 0.038 | 0.076 |
| JumpingJack | 16.38 | 17.02 | 0.045 | 0.089 |
| KneeKick | 7.36 | 7.77 | 0.054 | 0.102 |
| LegBack | 7.96 | 8.41 | 0.050 | 0.090 |
| LegCross | 6.60 | 6.68 | 0.042 | 0.073 |
| RonddeJambe | 22.59 | 22.65 | 0.057 | 0.103 |
| Running | 7.45 | 7.56 | 0.041 | 0.066 |
| Shuffle | 8.30 | 8.29 | 0.052 | 0.074 |
| SideLunges | 10.79 | 12.28 | 0.054 | 0.114 |
| SlowSkater | 10.29 | 10.71 | 0.054 | 0.113 |
| Squat | 12.94 | 13.54 | 0.048 | 0.112 |
| Average | 10.37 | 10.77 | 0.048 | 0.090 |

---

### 2) OOD-exercise（按运动独立训练 15 个模型）

|  | Our pose2emg | Official pose2emg | Our emg2pose | Official emg2pose |
|---|---:|---:|---:|---:|
| ElbowPunch | 18.11 | 21.43 | 0.070 | 0.080 |
| FrontKick | 10.05 | 11.53 | 0.082 | 0.097 |
| FrontPunch | 15.11 | 17.16 | 0.070 | 0.084 |
| HighKick | 14.75 | 16.87 | 0.127 | 0.133 |
| HookPunch | 18.60 | 17.80 | 0.084 | 0.092 |
| JumpingJack | 34.34 | 37.85 | 0.117 | 0.129 |
| KneeKick | 11.92 | 12.86 | 0.100 | 0.113 |
| LegBack | 14.46 | 15.96 | 0.097 | 0.105 |
| LegCross | 14.37 | 15.23 | 0.101 | 0.119 |
| RonddeJambe | 28.54 | 31.30 | 0.121 | 0.131 |
| Running | 12.58 | 12.60 | 0.063 | 0.072 |
| Shuffle | 13.26 | 14.27 | 0.071 | 0.076 |
| SideLunges | 23.63 | 24.78 | 0.125 | 0.137 |
| SlowSkater | 19.74 | 20.82 | 0.126 | 0.139 |
| Squat | 28.69 | 28.86 | 0.134 | 0.134 |
| Average | 18.54 | 19.96 | 0.099 | 0.109 |

---

### 3) OOD-person（leave-one-subject-out，独立训练 10 个模型）

|  | Our pose2emg | Official pose2emg | Our emg2pose | Official emg2pose |
|---|---:|---:|---:|---:|
| Subject0 | 25.34 | 26.15 | 0.090 | 0.096 |
| Subject1 | 23.34 | 23.72 | 0.105 | 0.114 |
| Subject2 | 13.29 | 14.88 | 0.091 | 0.100 |
| Subject3 | 16.22 | 17.91 | 0.098 | 0.108 |
| Subject4 | 15.84 | 16.27 | 0.085 | 0.102 |
| Subject5 | 14.92 | 15.96 | 0.091 | 0.100 |
| Subject6 | 17.68 | 17.88 | 0.104 | 0.109 |
| Subject7 | 16.06 | 16.72 | 0.099 | 0.109 |
| Subject8 | 10.80 | 14.66 | 0.106 | 0.115 |
| Subject9 | 17.21 | 17.48 | 0.110 | 0.112 |
| Average | 17.07 | 18.16 | 0.098 | 0.106 |

---

### 4) 全部平均值汇总（只写 Average）

|  | Our pose2emg | Official pose2emg | Our emg2pose | Official emg2pose |
|---|---:|---:|---:|---:|
| id | 10.37 | 10.77 | 0.048 | 0.090 |
| ood_exercise | 18.54 | 19.96 | 0.099 | 0.109 |
| ood_person | 17.07 | 18.16 | 0.098 | 0.106 |

---

### 结论摘要（只基于 Average）

- ID：Our 优于 Official
  - pose2emg：10.37 vs 10.77
  - emg2pose：0.048 vs 0.090
- OOD-exercise：Our 优于 Official
  - pose2emg：18.54 vs 19.96
  - emg2pose：0.099 vs 0.109
- OOD-person：Our 优于 Official
  - pose2emg：17.07 vs 18.16
  - emg2pose：0.098 vs 0.106
