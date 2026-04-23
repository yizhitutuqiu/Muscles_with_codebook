# 研究发现

**研究问题:** 如何实现 Stage 1 的时序对齐 Tokenizer，以提供显式的帧对齐 tokens（每帧 K=63），并附带时序上下文供 Stage 2 使用？同时如何优化算法，在不带 cond (`no_cond`) 的情况下，使 Stage 2 的平均误差降至 9-10 之间（优于 official_nocond 的 11.04）？

## 目前的理解

- 当前的 Stage 1 TCN Codebook (`tcn_codebook.yaml`) 使用自适应平均池化，将一段 30 帧的序列压缩成 1890 个 token。这种方式丢失了明确的“帧到 token”语义对齐。
- Stage 2 的设计假设了按帧的离散 token 输入。这与当前 Stage 1 的输出形状不兼容，因为 1890 个 token 并未在结构上对齐到每个单独帧的 63 个 token。
- `PLAN.md` 文档提出了 "Plan A: 帧对齐时序 Tokenizer" (Frame-Aligned Temporal Tokenizer) 方案：使用 1x1 卷积逐帧进行 token 扩展，从而保证 `token_count = 30 * 63`，并且索引符合 `token_index = t * 63 + k`。
- 根据 `now_results.csv`，目前的 `stage2(no_cond)` mean error 为 10.67，而 `official_nocond` 为 11.04。用户的最终目标是优化算法，使得 no_cond 下的 mean 误差进一步降低到 9-10 之间（甚至接近 9）。注入条件信息 (`cond`) 并非核心关注点。

## 模式与洞察

- 尚未进行实验。

## 经验教训与约束

- 所有的实验与验证都应在名为 `mia_custom` 的 tmux session 中运行，并保证实验的可复现性。
- 代码和文档的变更均需要使用 Git 进行版本管理。

## 开放性问题

- 1x1 卷积扩展在 Stage 1 的重构损失上与自适应池化相比，是否会带来显著影响？
- Stage 2 在接收这些具有时序上下文且重新对齐的 token 后，如何进一步修改其网络结构/损失函数以实现 10 以下的生成误差？

### 阶段 4：从底层架构入手优化无条件泛化 (H5)
- **挑战**: 在缺乏身份条件 (Identity Condition) 时，DCSA 注意力机制面临离散 Token 集合（63个）缺乏明确时空标识的问题，导致对齐过程严重依赖 Token 自身的语义，一旦序列长或动态复杂就容易泛化崩溃。同时，纯注意力机制 (DSTFormer) 对连续信号（如 EMG）的局部平滑度捕捉不足，且没有相对位置编码导致时序建模受限。
- **架构重构 (H5)**:
  - **离散 Token 时空编码注入**: 在 `stage2_pose2emg.py` 中，显式为 `z_disc` 注入与连续特征 `z_cont` 匹配的 `spatial_pe` 和 `temporal_pe`，为注意力计算提供强有力的空间和时间锚点，使其不再是一袋无序向量。
  - **增强型时空提取器 (DSTFormerV2)**: 引入了 `dstformer_v2.py`。在时间注意力分支中加入了 **旋转位置编码 (RoPE)** 以增强长序列的相对顺序建模；在时间 FFN 分支中引入了 **局部时序卷积 (LTC - Local Temporal Convolution)** 和 **SwiGLU 门控机制**，强制模型学习 EMG 连续信号的局部平滑性，并提升特征筛选能力。
- **批量消融**: 我们利用 `batch_train.stage2.yaml` 开启了 4 个并行的 GPU 实验 (包含 `H5 Full`, `H5 w/o LTC`, `H5 w/o RoPE`, 以及 `H2 Baseline`)，验证这些底层改动对降低 `no_cond` 误差的科学价值。
