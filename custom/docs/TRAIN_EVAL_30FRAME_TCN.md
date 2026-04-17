# 30 帧严格切片与 TCN 无状态 — 排查结论

本文档记录「训练/评测是否严格 30 帧一切片、是否存在跨切片时序或 TCN 状态未清空」的排查结论。

## 结论摘要

- **训练与评测均为「严格 30 帧一切片」**，无重叠、无随机 crop、无跨切片喂入同一 forward。
- **TCN 无跨 batch / 跨样本隐状态**，每次 `forward` 处理 N 个独立 30 帧片段，**无需也不存在「清空 TCN 状态」**。

因此不会出现「跨越多个切片跑时序」或「跑完一个切片没清空 TCN 导致信息串扰」的问题。

---

## 1. 数据侧：是否严格 30 帧一切片？

### 1.1 数据源

- **MIA**：`MyMuscleDataset` 每个 `__getitem__` 对应一个样本目录，返回该目录下 `joints3d.npy` / `emgvalues.npy` 等。
- 每个样本在 MIA 中通常为 **固定长度 30 帧**（一个 clip 一个目录）；若某样本 T>30，见下。

### 1.2 训练：`train_frame_codebook.py` 的 `_prepare_batch`

- **clip 模式**：`L = clip_len`（默认 30）。
- 若 `t > L`：只取 **前 L 帧**：`joints3d = joints3d[:, :L]`，`emg = emg[:, :, :L]`。
- 若 `t < L`：直接报错 `Clip length mismatch`。
- 输出形状：`(B, L*75)`（joints3d）、`(B, L*8)`（emg），即 **每条样本恰好 L 帧**。
- **无随机 crop**：始终「前 L 帧」，无滑动窗口、无随机起始点。

### 1.3 评测：`eval_frame_codebook_official_metrics.py` 的 `_prepare_eval_batch`

- 与训练一致：`L = clip_len`，`t != L` 时取 `joints3d[:, :L]`、`emg[:, :, :L]`。
- 同样 **无随机 crop**，无重叠切片。

### 1.4 是否「跨多个切片」喂入同一次 forward？

- **不会**。每次 `model(...)` 的输入是 **一个 batch**：B 条样本，每条 **独立** 的 30 帧（即 B 个独立 clip）。
- 没有「同一长序列被切成 0–29、30–59、60–89… 再在同一次或连续多次 forward 中按序喂入」的逻辑；DataLoader 每次取的是 **B 个不同样本目录**（或同一样本目录在不同 epoch 出现，但每次仍是同一段 30 帧）。
- 因此不存在「跨越多个切片跑时序」的情况。

---

## 2. 模型侧：TCN 是否有隐状态、是否需要清空？

### 2.1 结构

- 时序编码器：`TemporalConv1dEncoder` / `TemporalTCNEncoder` / `MixerTCNEncoder`。
- 实现均为 **Conv1d + GroupNorm（或类似）**，**无 LSTM/GRU/RNN**，无 `hidden`/`cell` 等跨步状态。

### 2.2 单次 forward 的语义

- 输入：`(N, in_dim)`，其中 `in_dim = seq_len * frame_dim`（如 30×75=2250）。
- 在内部 `view(n, seq_len, frame_dim)` 后沿时间维做 1D 卷积；**每次 forward 只看到当前这 N 个 clip 的 30 帧**。
- **不依赖** 上一 batch、上一时刻的任何「隐状态」；PyTorch 也不会在默认使用方式下在 batch 之间保留 TCN 的内部状态。

### 2.3 结论

- **TCN 无跨 batch / 跨样本的隐状态**。
- **不需要、也不存在「跑完一个切片清空 TCN」的操作**；每个 batch 的 forward 都是独立的。

---

## 3. 与「过拟合 / 哈希表背诵」的关系

- 你观察到的 **训练 loss 极低（~1e-8）、评测 ~0.3** 与「30 帧严格切片」「TCN 无状态」**不矛盾**。
- 当前实现已经保证：
  - 训练和评测都是 **严格 30 帧一切片**；
  - **没有** 随机 crop 导致 train 用「第 13–42 帧」、eval 用「第 0–29 帧」的错位；
  - **没有** 跨切片或跨 batch 的时序泄漏，也 **没有** 需要清空的 TCN 状态。
- 因此，train/eval 的差异更可能来自 **过拟合（如 TCN 对 30 帧离散序列的「哈希表式」记忆）**，而不是「切片或状态未清空」导致的技术 bug。改进方向（如你文中所提）：提高 codebook 利用率、时序抖动、适当减小 Decoder 容量等，仍适用于当前架构。

---

## 4. 代码中的明确约定（便于后续维护）

- **train**：`custom/train/train_frame_codebook.py` 中 `_prepare_batch` 的 clip 分支已注释：严格 30 帧一切片，B 个独立 clip，无随机 crop，TCN 无跨 batch 状态。
- **eval**：`custom/tools/eval_frame_codebook_official_metrics.py` 中 `_prepare_eval_batch` 的 clip 分支已注释：与 train 一致，严格 30 帧一切片，B 个独立 clip，TCN 无跨 batch 状态。
- **TCN**：`custom/models/temporal.py` 中 `TemporalConv1dEncoder` / `TemporalTCNEncoder` 的 docstring 已注明：无跨样本/跨 batch 隐状态，每次 forward 处理 N 个独立片段，无需 reset。

以上可作为「训练和评测是否严格 30 帧、是否有时序/状态错位」的正式结论与代码依据。
