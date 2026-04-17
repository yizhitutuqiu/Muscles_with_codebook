# Decoder 结构与 1e-8 悖论 — 深挖结论

本文档记录「为何训练 loss 可达 1e-8、评测却 ~0.3」的排查结论，并澄清：**当前实现并非「每帧一个 encoding、decoder 只看本帧」**。

---

## 0. 重要澄清：当前并不是「30 帧每帧一个 63*256、decoder 只看本帧」

你期望的设计是：**30 帧里每一帧出一个 encoding（例如每帧 1 个码或每帧 63*256），且 decoder 只看到本帧的量化结果**（逐帧独立，无跨帧混合）。

**当前实现实际是：**

- **Encoder**：`encoder_type: mixer_tcn`  
  - 输入是 **整段 clip**：`(B, 2250)` = (B, 30×75)，**一次** forward。  
  - **Mixer** 先把 2250 维 **整体** 线性投影成 63 个 token：`Linear(2250, 63*256)`，得到 (B, 63, 256)。  
  - **TCN** 再在这 63 个 token 的时间维上做卷积（这里的时间维是 Mixer 的 63，不是 30 帧）。  
  - 输出：**一个 clip 一个 63×256**，即 **整段 30 帧共享这 63 个 token**，不是「每帧出一个 63×256」。

- **VQ**：63 个 token 各量化成 1 个码 → **一个 clip 63 个码**。

- **Decoder**：`decoder_type: mixer` → `FourLayerMLPMixerDecoder`  
  - 输入：**整段** (B, 63×256)。  
  - 对 **63 个 token 做 token_mlp 混合**，所以 decoder 看到的是 **整段 63 维离散序列**，不是「只看到本帧」。

因此：**当前是「一个 clip 一个 63-token 序列，decoder 看整段 63」，而不是「每帧一个 encoding、decoder 只看本帧」**。若要做到你说的「每帧一个码、decoder 只看本帧」，需要改配置（见文末）。

---

## 1. 配置与代码结论（与假设 A/B 对照）

### 1.1 Decoder 实际结构（当前配置下）

- **配置**：`decoder_type: mixer`（joints3d / emg 均为 mixer）。
- **代码**：→ **`FourLayerMLPMixerDecoder`**（`mlp.py`）。
- **行为**：输入 (B, 63×256) → view(B, 63, 256) → 4×`_MLPMixerBlock`（**token_mlp 在 63 个 token 之间混合**）→ Linear → (B, 2250)。  
  → Decoder 看到的是 **整段 63 维离散序列**，不是 63 次独立的「一码一帧」。

**结论：在当前配置下，Decoder 带 63-token 全局混合，属于「假设 A」：可形成哈希表式背诵。**

---

### 1.2 是否存在特征泄露？（假设 B 下的三条）

- **残差/ Skip 连接**  
  - `_decode(mod, z_q)` 仅调用 `mod.decoder(z_q)`，只传入 `z_q`，没有把 Encoder 输出或原始输入接到 Decoder。  
  - **无泄露。**

- **量化维度**  
  - `_encode_quantize` 中：`z_e` shape 为 (B, token_count, code_dim)，flatten 成 (B*63, 256)，对 **每个 token 独立** 做 VQ，得到 63 个码。  
  - 没有在错误维度上做量化（不是「一帧 256 个码」）。  
  - **无误用。**

- **STE 与 Decoder 输入**  
  - `vq_ema.py` 中：`z_q_st = z_e + (z_q - z_e).detach()`，forward 值 = `z_q`（离散）。  
  - `_encode_quantize` 返回的正是该 `z_q_st`，以 `z_q` 变量名传入 `_decode`。  
  - **Decoder 收到的是离散的 z_q，STE 正确，无 z_e 泄露。**

---

## 2. 与「1e-8 vs 0.3」悖论的关系

- **为何训练能到 1e-8？**  
  Decoder 对 **63 个离散 token 整体** 做 Mixer 混合，相当于用「63 维离散密码」做键，容量足够时可以把训练集里的 (密码 → 精确 2250/240 维输出) 背下来，从而在训练集上逼近 1e-8。

- **为何 eval 是 ~0.3？**  
  评测时同一套密码序列可能因数据/归一化/随机性略有差异而错位一两个 token，或评测集密码未在训练中见过，Decoder 无法「查表」，只能依赖真实泛化，暴露出离散码本带来的量化误差 (~0.3)。

- **为何 emg_to_j3d 也能到 2.7e-7？**  
  同一逻辑：EMG 分支也输出 63 个离散码；Decoder(joints3d) 收到这 63 个码的「密码」后，同样可以背诵「这串 EMG 密码 → 对应 3D clip」，所以跨模态 loss 也能被背到极低。

---

## 3. 总结表

| 问题 | 结论 |
|------|------|
| Decoder 是否带时序/上下文？ | **是**：Mixer 对 63 个 token 做 token_mlp 混合，看到整段离散序列。 |
| 是否纯逐帧（假设 B）？ | **否**：不是「63 次独立 1 码→1 帧」映射。 |
| 是否存在残差/Skip 泄露？ | **否**：Decoder 只接收 z_q。 |
| 量化维度是否错误？ | **否**：63 token，每 token 1 码。 |
| STE 是否生效、Decoder 是否收到 z_q？ | **是**：forward 值为 z_q，无 z_e 泄露。 |
| 1e-8 的合理解释？ | **假设 A**：63 维离散密码 + Mixer 全局混合 → 哈希表式背诵。 |

建议方向（与之前一致）：提高 codebook 利用率、时序抖动、适当减小 Decoder 容量等，以削弱背诵、增强泛化。

---

## 4. 若要「每帧一个码、decoder 只看本帧」应如何配

目标：**30 帧 → 30 个 token（每帧 1 个）→ 30 个码 → decoder 每帧只根据本帧的码重建本帧**（无跨帧混合）。

- **Encoder**：必须 **按时间维输出 30 个 token**，不能先用 Mixer 把 30 帧压成 63 个 token。  
  - 做法一：`encoder_type: tcn`（或 `temporal_conv1d`），**不用** mixer_tcn。  
  - 配置：`token_count: 30`，`temporal.seq_len: 30`，`temporal.frame_dim: 75`，`temporal.pool: none`（30-in-30-out，不做 adaptive_avg 压成 63）。  
  - 这样 Encoder 输入 (B, 2250)，内部 view 成 (B, 30, 75)，TCN 沿时间维卷积后仍为 30 步，输出 (B, 30×256)。

- **Decoder**：必须 **逐帧、不跨帧混合**。  
  - 做法：`decoder_type: tcn`（或 `temporal_conv1d` / `temporal_tcn`）→ 使用 **TemporalConvDecoder**。  
  - 配置：`token_count: 30`，`temporal.seq_len: 30`，`temporal.frame_dim: 75`，`temporal.upsample: none`。  
  - TemporalConvDecoder 在 `upsample=none` 时不做插值，仅对每个 token 做 1×1 的 `to_frame`，即 **第 t 帧输出只依赖第 t 个 token**，decoder 只看本帧的量化结果。

- **注意**：当前 `mixer_tcn` 里传给 TCN 的 `temporal` 是 **seq_len=mix_tc（63）**，不是 config 里的 `temporal.seq_len: 30`（见 `frame_codebook.py` 中 `temporal_for_tcn = TemporalConfig(seq_len=mix_tc, ...)`）。所以只要用 mixer_tcn，时间维就是 63，不是 30。要「每帧一个」必须改成纯 TCN（或等价地 mixer 输出 30 token）且 decoder 用 TemporalConvDecoder + upsample=none。
