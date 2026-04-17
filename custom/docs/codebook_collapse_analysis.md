# Codebook 坍缩原因分析（warmup_2 中 loss 暴增、j3d_unique=1）

## 现象

- k-means 初始化后 warmup_2 前期 loss 正常（~1.0），j3d_unique≈30。
- 约 step 300→350 内 loss 突增到 ~8，j3d_ppl 降到 1，j3d_unique 降到 1（即所有 joints3d token 被量化到同一个 code）。

## 根因链

### 1. 未使用 code 的 EMA 会把它推向零向量

在 `vq_ema.py` 的 `_ema_update` 中：

- 每步对**本步未收到任何 assignment** 的 code：`cluster_size=0`，`embed_sum=0`。
- 因此 `ema_cluster_size *= decay`，`ema_embed_sum *= decay`（仅衰减、无新增）。
- 更新公式：`embedding = ema_embed_sum / smoothed`，且 `smoothed` 被 `clamp_min(eps)`。
- 当某 code 长期未被使用：`ema_cluster_size → 0`，`ema_embed_sum → 0`，从而 `smoothed` 被压到 eps，得到  
  `embedding = (0.99^t * center) / eps → 0`（t 为未使用步数）。

因此：**长时间未被使用的 code 的 embedding 会衰减到接近 0**。

### 2. warmup_2 下每步只有部分 code 被使用

- warmup_2 时 encoder/decoder 冻结，每 batch 只有 `batch_size * token_count`（如 8×30=240）个 token。
- codebook 有 256 个 code，每步至少约 16 个 code 得不到 assignment，每步都会衰减。
- 约 150 步后，大量 code 的 `ema_cluster_size` 和 `ema_embed_sum` 接近 0，**这些 code 的 embedding 被更成接近 0**。

### 3. 大量 code 变零 → 坍缩到单一 code

- 当多数 code 变成零向量时，对任意 `z_e`：到零向量的距离 ≈ `||z_e||^2`（都相同）。
- 少数仍被使用的 code 的 embedding 在 EMA 下会趋向当前 batch 的“中心”。
- 一旦有一个 code 的 embedding 接近“全局均值”，它会比零向量更接近绝大多数 `z_e`，于是**几乎所有 token 都选这个 code**。
- 结果：该 code 的 EMA 进一步被推向全局均值，其余 code 多为零、几乎不被使用 → **稳定在“1 个 code 通吃”** → j3d_unique=1，commitment loss 暴增。

### 4. dead-code reset 加剧重复与坍缩

- 在 step 200 等会执行 `_maybe_reset_dead_codes`，用**当前 batch 的 z_e 随机采样（有放回）** 替换 dead code。
- 当前 batch 的 240 个向量实际只对应约 30 个不同 code（j3d_unique=30），即来自约 30 个聚类。
- 有放回地抽 68 个替换 68 个 dead code → **大量替换后的 code 是相同或极相近的向量**（重复/近重复）。
- 等价于 codebook 里出现很多“重复行”，perplexity 容易进一步下降，与上面“多数 code 变零”一起，更容易在几步内坍缩到 1 个 code。

## 小结

| 因素 | 作用 |
|------|------|
| 未使用 code 的 EMA 衰减 | 将其 embedding 推成 0，为“多数为零、少数为中心”创造条件 |
| warmup_2 每步只用部分 code | 每步固定有 ~16+ code 得不到更新，持续衰减 |
| dead-code reset 有放回采样 | 用当前 batch 重采样产生大量重复 code，加速坍缩 |

## 修复方向

1. **防止未使用 code 变成零**：在 `_ema_update` 中，对 `ema_cluster_size` 低于某阈值的 code **不更新其 embedding**，保留当前值（例如保留 k-means 初始中心）。这样未使用的 code 不会变成零向量。
2. **降低 reset 带来的重复**：dead-code 重置时尽量**无放回采样**或对替换向量加小噪声，使新 code 彼此可区分，减少“多行相同”导致的坍缩。
3. **K-means 后同步 EMA 全局先验**：`ema_cluster_size.fill_(virtual)`、`ema_embed_sum = centers * virtual`（如 virtual=100），避免第一步被一个局部 batch 洗掉 k-means 中心。
4. **warmup_2 关闭 dead-code reset**：单 batch 里 j3d 有效簇远少于 256，用当前 batch 重置死码会破坏全局锚点；此阶段传 `skip_vq_dead_reset=True`。
5. **warmup_2 下 encoders 用 eval()**：`joints3d.eval()` / `emg.eval()`，避免 BN / online_std 在冻结阶段仍随 batch 漂移；VQ 保持 `train()` 以继续 EMA。
