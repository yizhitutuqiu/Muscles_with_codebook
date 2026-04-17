# 新的 Stage 1 (统一时序 Codebook) 设计方案

## 1. 背景与目标 (Background & Objective)
当前的 Stage 1（如 `tcn_codebook`）采用帧级或多 Token 级的量化策略：网络将 $T_{clip}$（例如30帧）经过 TCN 处理后，输出 $T \times K$ 个离散 Token，每个 Token 独立查询 Codebook 进行量化。
**新的设计目标**：将 $T_{clip}$ 帧作为一个不可分割的整体，通过网络编码为**1个统一的全局大 Encoding (Unified Encoding)**。这个统一的 Encoding 查询一个更大维度的 Codebook 完成量化，之后由对称的 Decoder 结构将其解压回完整的 $T_{clip}$ 帧序列。这种设计强制 Codebook 学习整个 Clip 的全局时序和空间相关性，从而可能提取到更加高维和结构化的运动基元。

## 2. 算法方案设计 (Algorithm Design)

### 2.1 输入表示
- **输入数据**：长度为 $T_{clip}$ 帧，每帧特征维度为 $F$（例如 30帧 $\times$ 75维，Flatten 后总维度为 2250）。

### 2.2 统一编码器 (Unified Clip Encoder)
我们推荐基于 TCN (Temporal Convolutional Network) 的方案，计算高效且与当前架构兼容性最好。
1. **Reshape**：将 `(N, 2250)` 还原为时序结构 `(N, F, T_{clip})`。
2. **特征提取**：经过多层膨胀一维卷积 (Dilated Conv1d + GroupNorm + GELU)，提取时序上下文，输出形状为 `(N, D_{hidden}, T_{clip})`。
3. **全局时序池化**：使用 `AdaptiveAvgPool1d(1)` 将时间轴 $T_{clip}$ 压缩为 1，得到全局特征 `(N, D_{hidden}, 1)`。
4. **降维映射**：通过 `Conv1d(D_{hidden}, D_{code}, 1)` 映射到统一的大 Encoding 维度，输出 `(N, D_{code}, 1)`。
5. **输出**：最终 Reshape 为 `(N, D_{code})`，即为代表整个 $T_{clip}$ 运动片段的唯一 Token。

### 2.3 向量量化 (Vector Quantization)
- 此时 `token_count = 1`。
- 因为一个 Token 需要表征整个片段的运动细节，必须大幅增加单条 Code 的维度（例如 `code_dim = 1024` 或 `2048`）以及 Codebook 的容量（例如 `num_codes = 4096` 或 `8192`）。
- 该全局 Token 直接计算与 Codebook 中各个条目的 L2 距离，完成最近邻量化替换。

### 2.4 统一解码器 (Unified Clip Decoder)
1. **扩展时序**：输入量化后的全局 Encoding `(N, D_{code})`，将其 Reshape 为 `(N, D_{code}, 1)`，然后沿时间轴复制（Repeat）$T_{clip}$ 次，得到 `(N, D_{code}, T_{clip})`。
2. **注入位置编码 (Positional Encoding)**：由于所有帧被赋予了相同的初始特征，必须加入**可学习的时序位置编码** `(1, D_{code}, T_{clip})`，为网络提供每帧相对位置的先验信息。
3. **特征解码**：将叠加了位置编码的特征输入到对称的 TCN Blocks 中进行逐步解码与特征融合。
4. **映射回原空间**：最后通过 `Conv1d(D_{hidden}, F, 1)` 映射回原始的单帧特征维度，输出 `(N, F, T_{clip})`。
5. **输出**：Reshape 回 `(N, 2250)` 即可计算重建 Loss。

---

## 3. 代码方案设计 (Code Implementation Plan)

遵循**最小侵入原则 (Minimal Invasiveness)**，我们不需要修改任何外层的训练循环、Loss 计算或数据流逻辑。当前 `FrameCodebook` 的架构天然支持自定义的 `token_count` 和 `code_dim`，只需在模型库中新增编解码器类并在配置中注册即可。

### Step 1: 新增 Clip 级时序模型类
在 `/data/litengmo/HSMR/mia_custom/custom/models/temporal.py` 中，新增如下两个类：
- **`ClipUnifiedTCNEncoder`**：
  核心实现逻辑：输入重塑为 `(N, F, T)` $\rightarrow$ `TCN Blocks` $\rightarrow$ `F.adaptive_avg_pool1d(x, 1)` $\rightarrow$ `Conv1d(proj)` $\rightarrow$ 展平为 `(N, code_dim)`。
- **`ClipUnifiedTCNDecoder`**：
  核心实现逻辑：输入 `(N, code_dim)` $\rightarrow$ 重塑并 `x.unsqueeze(-1).repeat(1, 1, T)` $\rightarrow$ 加上 `nn.Parameter` 位置编码 $\rightarrow$ `TCN Blocks` $\rightarrow$ `Conv1d(proj)` $\rightarrow$ 展平为 `(N, T * F)`。

### Step 2: 在工厂/配置解析中注册新类型
在 `/data/litengmo/HSMR/mia_custom/custom/models/frame_codebook.py` 中：
找到解析 `encoder_type` / `decoder_type` 的地方（约 123 行附近），添加对新类型的支持：
```python
if encoder_type in ("clip_unified_tcn", "unified_tcn"):
    self.encoder = ClipUnifiedTCNEncoder(
        in_dim=mod_cfg.in_dim,
        hidden_dim=mod_cfg.hidden_dim,
        code_dim=mod_cfg.code_dim,
        temporal=temporal_cfg
    )
# Decoder 同理注册 ClipUnifiedTCNDecoder
```

### Step 3: 创建对应的训练配置文件
新建 `/data/litengmo/HSMR/mia_custom/custom/configs/clip/unified_clip_codebook.yaml`，核心修改项如下：
```yaml
model:
  vq:
    num_codes: 8192      # 更大的字典容量以涵盖全局片段组合
    code_dim: 1024       # 更长的 Encoding 维度
  modalities:
    joints3d:
      in_dim: 2250       # 30 * 75
      hidden_dim: 1024
      token_count: 1     # 【核心修改】只输出1个统一的Token
      code_dim: 1024
      encoder_type: unified_tcn
      decoder_type: unified_tcn
      temporal:
        seq_len: 30
        frame_dim: 75
        kernel_size: 3
        num_layers: 4
```

通过这三步，我们可以在完全不破坏原有逻辑的前提下，优雅地切换到基于全局片段统一量化的新 Stage 1 架构。