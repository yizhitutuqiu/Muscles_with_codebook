## 目标与约束

目标：让 `custom/configs/clip/tcn_codebook.yaml` 这套“时序 codebook（Stage1 用 30 帧 TCN 编码）”与 Stage2 兼容，同时保留“每一帧的离散 token 带有时序上下文”的信息。

关键约束（按需求理解）：

- Stage2 侧希望仍然以“单帧粒度”的离散 token 作为输入语义（即每帧都有自己的一组 token），但这些 token 不是逐帧独立编码出来的，而是来自 Stage1 的时序网络（看过整段序列后产生的、包含时序信息的每帧 token）。
- Stage1/Stage2 都允许修改；如果当前 Stage1 的 token 不可解释地对应到 30 帧中每一帧，也允许重设计 Stage1 的 tokenization，使其对齐到“每帧 K 个 token”。

本文档给出一个谨慎的兼容方案设计：先把 token 的“帧对齐语义”做扎实，再让 Stage2 能消费两种 Stage1（逐帧/时序帧对齐）而不改训练脚本使用方式。

---

## 现状与不兼容点

### Stage1（tcn_codebook.yaml）的现状

该配置是 clip 级别输入：

- joints3d: `in_dim = 2250 = 30 * 75`
- emg: `in_dim = 240 = 30 * 8`
- token_count: `1890 = 30 * 63`
- encoder_type: `tcn`
- decoder_type: `frame_mixer (tokens_per_frame=63)`

当前 `TemporalTCNEncoder` 的行为（见 `custom/models/temporal.py`）：

- 先把 `(N, seq_len*frame_dim)` reshape 为 `(N, frame_dim, seq_len)`，沿时间做 Conv1d
- 得到 `(N, code_dim, seq_len)`
- 然后依据 `temporal.pool` 做 pooling 到 `token_count`

在 `tcn_codebook.yaml` 里 `temporal.pool: adaptive_avg` 且 `token_count=1890`，这会触发 `adaptive_avg_pool1d(y, output_size=1890)`，也就是把长度 30 的时序特征“重采样/池化”到长度 1890。这种 tokenization：

- 虽然形状上能工作，但并没有清晰的“第 t 帧对应哪 63 个 token”的语义对齐
- 很难在 Stage2 里做到“单帧消费，但 token 带时序信息”的可解释分组

### Stage2 的现状

Stage2 当前离散分支假设 Stage1 是逐帧 codebook：

- Stage2 传入 `(B,T,25,3)`，每帧 reshape 成 `(B*T, 75)`，再调用 `stage1.joints3d.normalize/encoder/vq` 得到 `(B*T, 63, dim)`
- 因此与 `tcn_codebook.yaml` 这种 `in_dim=2250` 的 Stage1 会直接 shape mismatch

---

## 设计总览：把 Stage1 的 token 做到“帧对齐 + 时序上下文”，Stage2 消费“每帧 token”

核心思想：

1. Stage1 仍然看整段 clip（30 帧）并通过 TCN 注入时序上下文；
2. 但 Stage1 的输出 token 必须具有明确的帧分组：每帧固定 `K=63` 个 token，且 token 的索引能无歧义映射到 `(t, k)`；
3. Stage2 的离散分支输入仍然按帧组织：`z_disc_bt: (B, T, K, dim)`，只是这些 token 是从 Stage1 的“整段编码”中 reshape 出来的，因此每帧 token 带上下文信息；
4. Stage2 在实现上同时支持两类 Stage1：
   - 逐帧 Stage1（`in_dim=75, token_count=63`）
   - 时序帧对齐 Stage1（`in_dim=2250, token_count=30*63` 且有帧对齐 tokenization）

---

## 方案 A（推荐）：Stage1 新增“Frame-Aligned Temporal Tokenizer”

### A1. Stage1：让 token_count=30*63 具有明确的帧分组语义

目标：对 joints3d/emg 两个模态，Stage1 encoder 输出的 token 序列满足：

- `token_count = seq_len * tokens_per_frame`
- token 索引规则：`token_index = t * tokens_per_frame + k`
- reshape 后：`(B, token_count, dim)` → `(B, T, tokens_per_frame, dim)`

实现策略（在 encoder 内完成，而不是靠 adaptive pooling “凑长度”）：

1. TCN backbone 仍然产生按帧对齐的特征：
   - 输入 `(N, seq_len*frame_dim)` → reshape → `(N, frame_dim, seq_len)`
   - Conv1d/TCN blocks → `(N, hidden_dim, seq_len)`
   - 1x1 conv 投到基础特征维：`to_code: (N, code_dim, seq_len)`

2. 用一个“token expansion”的 1x1 conv 在每个时间步产生 `tokens_per_frame` 个 token：
   - `expand: Conv1d(code_dim, code_dim * tokens_per_frame, kernel_size=1)`
   - 输出 `(N, code_dim * K, seq_len)`
   - reshape → `(N, seq_len, K, code_dim)` → flatten → `(N, seq_len*K, code_dim)`
   - 最终 view 成 `(N, (seq_len*K)*code_dim)` 与现有 Stage1 API 兼容

3. 关闭 `adaptive_avg` 这类会打乱帧语义的 pool：
   - 新增一种 pool 模式：`temporal.pool: frame_tokens`
   - 并新增 `temporal.tokens_per_frame: 63`

这样做的好处：

- Stage1 的 token 对齐是结构性保证，而不是靠 pool 的“插值/重采样”
- Stage2 可以可靠 reshape 出 `(B,T,63,dim)`，满足“每帧离散 token 带时序信息”
- 仍可复用现有 `frame_mixer` decoder：它天然就是“每帧用同一个 decoder 解码”

### A2. Stage1 配置改法（以 tcn_codebook.yaml 为例）

在 `custom/configs/clip/tcn_codebook.yaml` 里建议改成：

- `modalities.{joints3d,emg}.temporal.pool: frame_tokens`
- `modalities.{joints3d,emg}.temporal.tokens_per_frame: 63`
- `token_count` 仍然保持 `1890`
- decoder 仍然用 `frame_mixer.tokens_per_frame: 63`

注意：这意味着 Stage1 的 token_count=1890 真的等于 `30*63`，且每帧的 63 个 token 是 encoder 直接生成的。

### A3. Stage1 需要改的代码点（最小侵入）

文件：`custom/models/temporal.py`

建议改动：

1. `TemporalConfig` 增加字段：
   - `tokens_per_frame: Optional[int] = None`

2. `TemporalTCNEncoder.forward()` 对 `temporal.pool` 增加分支：
   - `adaptive_avg`（保留，用于旧配置）
   - `none`（保留，frame-aligned: token_count == seq_len）
   - `frame_tokens`（新增，要求 token_count == seq_len * tokens_per_frame）

3. `TemporalTCNEncoder.__init__()` 增加 `expand` 层，仅在 `pool=frame_tokens` 时启用：
   - `self.expand = nn.Conv1d(code_dim, code_dim * tokens_per_frame, 1)`
   - forward 中先 `y = self.to_code(...)`，再 `y = self.expand(y)`，再 reshape 到 `(N, token_count, code_dim)`

同样逻辑需要在 `TemporalConv1dEncoder`（如果未来也想用 conv1d variant）里镜像实现；但当前配置用的是 `tcn`，可以先只做 `TemporalTCNEncoder`。

### A4. Stage2：离散分支支持“时序帧对齐 Stage1”

文件：`custom/stage2/models/stage2_pose2emg.py`

现状 `_stage1_discrete_tokens(joints3d_flat: (B*T,75))` 假设逐帧 Stage1。

建议改成两条路径（自动判断，不让训练脚本的调用方式变化）：

1. 若 `stage1.joints3d.cfg.in_dim == 75`：
   - 走旧逻辑：逐帧 encode+vq
   - 得到 `z_disc: (B*T, 63, dim)` → reshape 成 `(B,T,63,dim)`

2. 若 `stage1.joints3d.cfg.in_dim == 2250`：
   - 走 clip 逻辑：输入 `joints3d: (B,T,25,3)` flatten 成 `x_clip: (B, 2250)`
   - `x_n = stage1.joints3d.normalize(x_clip, update=False)`
   - `z_e = stage1.joints3d.encoder(x_n)` → view `(B, token_count, dim)`，其中 token_count 预期 1890
   - `z_q, idx = stage1.vq(z_e.reshape(B*token_count, dim))`
   - reshape `z_q` 为 `(B, token_count, dim)`，再 reshape 成 `(B, T, tokens_per_frame, dim)`
     - tokens_per_frame 可从 config 推断：`tokens_per_frame = token_count // T`（要求能整除，并在代码里 assert）
   - 输出给 Stage2 融合层时使用 `(B,T,63,dim)`

同时 `idx_j3d` 的输出也应改成兼容两种情况：

- 逐帧 Stage1：`idx_j3d: (B,T,63)`
- 时序帧对齐 Stage1：`idx_j3d: (B,T,63)`（从 `(B,1890)` reshape 得到）

### A5. Stage2 的 residual 模式如何处理

Stage2 residual 模式当前使用 Stage1 的 `emg.decoder(z_disc_flat)` 作为保底：

- 对逐帧 Stage1：`z_disc_flat` 是 `(B*T, 63*dim)`，decoder 输出 `(B*T, 8)` 再 view `(B,T,8)`
- 对时序帧对齐 Stage1：Stage1 的 emg decoder 是 clip 级别（`in_dim=240`），因此：
  - 应使用整段 `z_disc_clip_flat: (B, 1890*dim)` 喂给 `stage1.emg.decoder` 得到 `(B, 240)`，再 reshape 成 `(B, T, 8)`
  - 不能只拿单帧 63 token 去解码（否则维度不对，也不符合 Stage1 的训练方式）

因此：residual 模式下 Stage2 需要同时保留两种离散 token 形式：

- `z_disc_bt: (B,T,63,dim)`：用于融合
- `z_disc_clip_flat: (B, 1890*dim)`：用于 Stage1 emg base（仅 residual 模式需要）

---

## 方案 B（可选）：支持“真正单帧流式输入”的时序 token（Causal + 缓存）

如果真实需求是流式（online）：

- 时刻 t 的 token 只能依赖历史（≤t），不能偷看未来帧

那么需要把方案 A 的 TCN 改成 causal（或至少在推理时只用 past window），并在 Stage2 侧维护一个长度 30 的 buffer：

1. Stage1：把 `_TemporalBlock` padding 改成 causal padding：
   - 当前是对称 padding（看未来），改成只在左侧 pad
   - forward 后裁剪回原始长度

2. Stage2：新增一个 streaming wrapper：
   - 维护 `last_30_frames`（joints3d）ring buffer
   - 每来一帧，用 buffer 组成 `(1,30,25,3)`，跑 Stage1 得到当前帧的 63 token
   - 再把这一帧 token 喂给下游（若 Stage2 也要流式，需要把 DSTFormer 换成可流式的 temporal 模块，或做 chunk 推理）

这个方向实现成本明显高于方案 A（尤其是 Stage2 的 DSTFormer 流式化），因此建议先落地方案 A（offline，但每帧 token 带上下文），等指标跑通后再考虑流式。

---

## 与现有训练/评估脚本的兼容策略

### Stage1 训练（train_frame_codebook.py）

- 保持训练入口不变（仍然以 clip pack 模式喂 `(B, 2250)` / `(B,240)`）
- 只要 encoder 输出的 token_count 与 cfg 一致，VQ/decoder 代码都能工作
- 需要在 config 中显式声明 `temporal.pool: frame_tokens` 与 `tokens_per_frame`

### Stage2 训练（train_stage2_pose2emg.py）

- 训练输入仍是 `(B,T,25,3)`，Stage2 自己内部决定如何调用 Stage1
- 只要 Stage2 离散分支实现了“若 Stage1 是 clip 输入则按 clip 路径编码并 reshape 成每帧 token”，训练脚本无需改

---

## 风险点与排雷清单

1. “pool=adaptive_avg 输出长度>输入长度” 的旧行为会产生不稳定/不可解释 token，对齐不到帧；避免继续依赖它来做 1890 token。
2. token_count 与 (seq_len, tokens_per_frame) 的一致性要强校验：
   - `token_count == seq_len * tokens_per_frame`，否则 Stage2 reshape 会错
3. Stage1 的 OnlineStandardizer：
   - clip 输入时 standardizer 的 `in_dim` 是 2250/240；Stage2 复用时要确保 normalize 的输入维度匹配（方案 A 的 Stage2 clip 路径是匹配的）
4. residual 模式 base emg 的维度：
   - 必须用整段 token 解码（B,1890*dim → B,240 → B,T,8）
5. 如果未来想做真正单帧推理（online），需要把 TCN causal 化；否则 token 会偷看未来帧，离线评估没问题但在线不可用。

---

## 最小验收标准（建议实现后用这些自检）

1. Stage1（tcn_codebook）单 batch forward：
   - joints3d 输入 `(B,2250)` 输出 `z_e/z_q` reshape 成 `(B,30,63,dim)` 不报错
   - `frame_mixer` decoder 输出 `(B,2250)` 且 reshape 成 `(B,30,75)`
2. Stage2 离散分支：
   - 同一个 Stage1 ckpt：
     - 能输出 `z_disc_bt: (B,30,63,dim)` 与 `idx_j3d: (B,30,63)`
3. residual 模式：
   - `stage1.emg.decoder(z_disc_clip_flat)` 输出可 reshape `(B,30,8)`

---

## 实施步骤（建议顺序）

1. 改 Stage1：在 `TemporalTCNEncoder` 增加 `frame_tokens` 模式与 `tokens_per_frame`，替换 `adaptive_avg` 的 1890 token 生成方式。
2. 改 Stage1 config：把 `tcn_codebook.yaml` 的 `temporal.pool` 改为 `frame_tokens`，补齐 `tokens_per_frame: 63`。
3. 改 Stage2：离散分支支持两种 Stage1（逐帧 vs clip 帧对齐），并处理 residual base 的整段解码。
4. 跑最小 shape 自检（无需训练）：
   - 构造随机 joints3d `(B,30,25,3)`，检查 Stage2 forward 的所有 shape 与 reshape 规则正确。
5. 再做小规模训练/评测对比，验证“时序 token”的收益是否存在（否则考虑增加 Stage1 的因果性或增大感受野）。
