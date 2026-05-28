## Dual DCSA（双向 Discrete–Continuous Spatial Attention）设计与原理说明

本文档解释我们在 Stage2 中使用的 **双向 DCSA**（配置项 `fusion_type: symmetric_dcsa`）的设计动机、张量形状、计算流程与实现细节，并说明它与单向 DCSA（`dcsa_asymmetric`）的差异与适用场景。

相关实现位置：

- 双向模块实现： [dcsa.py](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L100-L162)
- 单向模块实现： [dcsa.py](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L16-L63)
- 融合模块工厂： [fusion.py](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/fusion.py#L69-L100)
- Stage2 调用融合并送入时序主干： [stage2_pose2emg.py](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/stage2_pose2emg.py#L421-L474)

---

### 1. 背景：为什么需要 Discrete–Continuous 融合

Stage2 同时拥有两种信息来源：

- **连续特征（continuous tokens）**：来自连续编码器（例如 `joint_25` 或 `emg_8`），刻画细粒度时序变化与局部细节。
- **离散先验（discrete tokens）**：来自 Stage1 codebook（VQ-VAE 风格），刻画“动作模式/片段语义”的可迁移结构性先验。

融合的目标不是简单拼接，而是实现两者之间的 **信息交互（message passing）**：

- 连续分支借助离散先验获得更强的结构约束与泛化能力；
- 离散先验借助连续分支获得与当前样本细节一致的对齐与校正（尤其在跨主体/跨动作分布偏移时）。

---

### 2. 名称与“Spatial vs Temporal”的准确含义

我们这里的 DCSA 全称在代码里写为 **Discrete–Continuous Spatial Attention**：[dcsa.py:L16-L29](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L16-L29)

其中 **Spatial** 指 attention 的“序列维”主要是 **token 维**（例如关节 token 或离散 token），而不是时间维 `T`：

- 输入形状是 `(B, T, N, C)`，其中 `N` 表示“同一帧内的 token 集合大小”
- 实现里会 reshape 为 `(B*T, N, C)` 后做 attention，因此 attention 的交互发生在“每帧内部的 token 集合”上

时间维建模在 DCSA 之后由时序主干完成（DSTFormer / 官方 TransformerEnc / TCN 等）。

---

### 3. 记号与张量形状

记号定义：

- batch 大小：`B`
- 序列长度（帧数）：`T`（通常为 30）
- continuous tokens 数量：`N_c`（例如 joints3d: 25；emg: 8 或 1，取决于连续编码器）
- discrete tokens 数量：`N_d`（由 Stage1 产生并在 Stage2 对齐/重复后的 token 数）
- token 通道维：`C`（例如 256；或对齐实验中的 126）

两路输入：

- `X_c ∈ R^{B×T×N_c×C}`：连续分支 tokens
- `X_d ∈ R^{B×T×N_d×C}`：离散先验 tokens

---

### 4. 单向 DCSA（baseline attention fusion）

单向 DCSA（`DiscreteContinuousSpatialAttention`）对应代码中的最小实现：[dcsa.py:L16-L63](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L16-L63)

它的核心是 **continuous 作为 Query，discrete 作为 Key/Value**：

1) reshape 到每帧独立计算：

- `Q = reshape(X_c) ∈ R^{(B·T)×N_c×C}`
- `K = V = reshape(X_d) ∈ R^{(B·T)×N_d×C}`

2) cross-attention：

- `A_c = Attn(Q, K, V) ∈ R^{(B·T)×N_c×C}`

3) 残差 + LayerNorm：

- `Y_c = LN(X_c + Drop(A_c)) ∈ R^{B×T×N_c×C}`

输出 token 数量与 continuous 相同（`N_c`），离散 token 不再继续向下游传播。

适用场景：

- 更轻量、参数/计算更省
- 强行保证“下游 token 结构不变”（只保留 continuous token）

---

### 5. 双向（Dual / Symmetric）DCSA：设计要点

双向 DCSA（`SymmetricDCSA`）对应实现：[dcsa.py:L100-L162](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L100-L162)

它的关键思想是：**同时更新 continuous 和 discrete 两个 token 集合**，即进行双向的信息交换：

- continuous 通过离散先验进行调制：`X_c ← f(X_c, X_d)`
- 离散先验通过连续细节进行校正：`X_d ← g(X_d, X_c)`

这种“双向更新”在跨域时通常更稳定：离散 token 不再是静态先验，而是可被样本细节对齐的动态先验。

---

### 6. 双向 DCSA 的计算流程（逐步）

#### Step 0：按帧展开

实现里同样按帧做 attention，先 reshape：

- `S = reshape(X_c) ∈ R^{(B·T)×N_c×C}`（代码里变量 `f_s`）
- `Q = reshape(X_d) ∈ R^{(B·T)×N_d×C}`（代码里变量 `f_q`）

#### Step 1：共享隐空间投影（可学习）

代码中使用两个线性层把两路映射到同一空间（维度仍为 `C`）：[dcsa.py:L115-L117](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L115-L117)

- `S' = W_s S`
- `Q' = W_q Q`

这一步的意义：

- 允许两路在 attention 前进行可学习的对齐（例如不同分布、不同统计量）
- 让 cross-attention 更接近“跨模态对齐”的典型建模方式

#### Step 2：双向 cross-attention（核心）

两次 cross-attention：[dcsa.py:L147-L152](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L147-L152)

- **C → D（continuous updated by discrete）**
  - `A_{c←d} = Attn(Q=S', K=Q', V=Q') ∈ R^{(B·T)×N_c×C}`
- **D → C（discrete updated by continuous）**
  - `A_{d←c} = Attn(Q=Q', K=S', V=S') ∈ R^{(B·T)×N_d×C}`

注意：这里的方向命名容易混淆。更严格地说：

- 第一条更新的是 continuous token（Query 来自 continuous）
- 第二条更新的是 discrete token（Query 来自 discrete）

#### Step 3：反向投影回原空间

对应代码：[dcsa.py:L153-L156](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L153-L156)

- `ΔS = W'_s A_{c←d}`
- `ΔQ = W'_q A_{d←c}`

#### Step 4：残差融合 + LayerNorm

对应代码：[dcsa.py:L157-L159](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L157-L159)

- `Ŝ = LN_s(S + Drop(ΔS))`
- `Q̂ = LN_q(Q + Drop(ΔQ))`

再 reshape 回 `(B,T,*,C)`。

#### Step 5：token 维 concat（输出形态）

最后把两路更新后的 tokens 在 token 维拼接：[dcsa.py:L161-L162](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/dcsa.py#L161-L162)

- `Y = concat([Ŝ, Q̂], dim=token) ∈ R^{B×T×(N_c+N_d)×C}`

这一步的动机：

- 让下游模块（例如 `emg_head` 或自研时序主干）能同时“看到”两路更新后的 token 集合
- 提供更高的表示容量：token 数量增加，相当于显式保留了离散/连续两种表征

在 Stage2 里，token 数量的变化也被显式用于 `emg_head` 的配置（`fused_token_count`）：[stage2_pose2emg.py:L171-L179](file:///data/litengmo/HSMR/mia_custom/custom/stage2/models/stage2_pose2emg.py#L171-L179)

---

### 7. 为什么双向会更强（直觉 + 机制）

从机制上看，双向 DCSA 相比单向多了一个关键能力：

- 单向：离散 token 只作为 KV 提供信息，离散本身不被更新；它更像“静态先验字典”。
- 双向：离散 token 也会被 continuous tokens 更新（`D → C` 分支），离散先验变为“可被当前样本校正的动态先验”。

因此在分布偏移（OOD-exercise / OOD-person）时，双向更容易通过样本细节对齐离散表示，减少“先验不匹配”的影响。

---

### 8. 与官方 TransformerEnc 主干结合时的注意事项

当时序主干是官方 TransformerEnc（其输入形状是 `(B,T,126)`，没有 token 维）时：

- 若直接把 `Y ∈ R^{B×T×(N_c+N_d)×C}` 送入主干，会破坏其输入接口
- 因此需要额外的聚合/映射策略（例如 token-wise mean、attention pooling、MLP projection）把多 token 还原为单 token 形式

这也是我们在不同实验设定中引入 `mean over tokens` 或 “MLP to token” 的原因（属于“接口适配”，不是 DCSA 本身的一部分）。

---

### 9. Ablation 建议：如何严谨比较“单向 vs 双向”

做 DCSA ablation 时，为了确保对比公平，建议固定：

- Stage1 codebook/encoder
- continuous encoder 类型与 token 数（`N_c`）
- 时序主干（`temporal_type`）与 head
- 训练超参（batch/steps/lr）

只切换融合模块：

- `fusion_type: symmetric_dcsa`（双向 + concat）
- `fusion_type: dcsa_asymmetric`（单向，输出 token 数保持 `N_c`）

注意：双向输出 token 数不同（`N_c+N_d`），下游 head 的输入 token 数也随之变化；这属于双向机制带来的“表示容量变化”，是对比的一部分，通常不应人为强行抹平。

---

### 10. 论文/图中建议写法（不引起歧义）

推荐用更中性的学术表述：

- “We employ a **bidirectional discrete–continuous cross-attention module** to exchange information between discrete codebook tokens and continuous feature tokens.”
- “The module updates both token sets via two cross-attention passes (continuous→discrete and discrete→continuous), followed by residual connections and LayerNorm.”

如果要点出 concat：

- “We concatenate the updated token sets along the token dimension to form a joint token sequence for downstream processing.”
