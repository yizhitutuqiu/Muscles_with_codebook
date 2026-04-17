# Condition (cond) 引入 Stage 2 的可行性分析与修改方案

## 1. 现状分析

在官方的 `MIADatasetOfficial` 中，`cond`（即 `condval`）是一个表示人物身份的一维浮点标量（例如 Subject0 为 `[1.0]`, Subject1 为 `[0.9]`，未见过的视为 `[0.1]`）。
在官方模型中，这个 `cond` 会通过一些映射或者拼接，注入到网络中作为辅助信息，指导肌肉活动的预测。

当前我们自己复现的 `Stage2Pose2EMG` 模型中，输入参数只有 `joints3d`，没有接收和使用 `cond`。

## 2. 引入 `cond` 的可行性

**完全可行，且非常简单。**

我们遵循**最小侵入原则**（Minimal Invasion Principle），在当前架构中引入 `cond` 的最佳方式是将这个一维的标量映射为与模型隐藏层维度（`dim=256`）一致的特征向量，然后将其作为全局上下文（Global Context）与 `joints3d` 的特征进行融合。

由于我们在 Stage 2 的核心融合模块之后，得到的是一个形状为 `(B, T, N, C)` 的特征张量 `z_fused`，我们可以将 `cond` 特征注入到这里，或者在送入 EMG Head 之前注入。

## 3. 代码修改方案

### 3.1 方案 A：特征加和（当前推荐的“最小侵入”方案）

如上文所述，将 `cond` 映射为 `dim=256` 的特征后，在 `z_fused` 阶段进行相加（`z_fused = z_fused + cond_feat`）。
- **优点**：符合 Transformer/Mixer 架构的常规设计（类似 Positional Encoding 的注入方式），不改变任何现有特征流的形状，能够平滑兼容无条件模型。

### 3.2 方案 B：官方原汁原味的拼接法（Concat）

在官方的 `TransformerEnc` 模型（`musclesinaction/models/modelposetoemg.py`）中，`cond` 的注入方式非常直接粗暴：
```python
        # src.shape: (B, 30, 126)  (经过 CNN 提取的骨骼特征 + 位置编码)
        condition = torch.ones(src.shape[0], src.shape[1], 2).to(self.device)
        condition = condition * condval.reshape(condval.shape[0], 1, 1)
        
        # 直接拼接：126维骨骼特征 + 2维的 condition = 128 维特征
        srccat = torch.cat([src, condition.type(torch.cuda.FloatTensor)], dim=2)
        
        # 128维送入 Transformer Encoder
        transformer_out = self.transformer0(src)
```

官方的做法是：**人为地将网络主干的维度设为 128，骨骼特征占 126 维，然后在最后 2 维直接暴力拼接上复制了两遍的 `condval`。**

如果我们要在当前的 `Stage2Pose2EMG` 中“尽可能接近”官方做法，可以在特征送入时序网络之前（或者在连续分支的输入端）采用拼接（Concat）的方式。例如，在送入时序模块 `self.temporal` 之前，把 `cond` 拼在 `z_fused` 的特征维度上，然后再用一个 Linear 压回 `256` 维：
```python
    def forward(self, joints3d: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # ...
        z_fused = self.fusion(z_cont_bt, z_disc_bt)
        
        if self.use_cond_concat and cond is not None:
            # cond: (B, 1) -> expand -> (B, T, N, 1)
            cond_expand = cond.view(b, 1, 1, 1).expand(b, t, n, 1)
            # Concat 后维度变成 257
            z_fused_cat = torch.cat([z_fused, cond_expand], dim=-1)
            # 压回 256 维
            z_fused = self.cond_proj(z_fused_cat) 
```

### 3.3 方案对比与结论

| 特性 | 方案 A（特征相加） | 方案 B（官方拼接法） |
| :--- | :--- | :--- |
| **代码侵入性** | **极低**（仅一个广播加法，不改变形状） | 较高（改变了张量通道数，需要额外的 Linear 降维） |
| **官方一致性** | 逻辑一致（作为全局条件注入），实现不同 | **完全对齐**（都是通过通道拼接的方式强塞标量） |
| **可扩展性** | 极强（未来如果 cond 变成高维 Embedding，只需改 MLP 输入） | 较差（如果 cond 维度变大，Concat 会导致通道数剧增） |
| **对时序模块的影响**| 零影响，特征平滑融合 | Concat 后必须立刻过一层 Linear 混合，否则后续的 Self-Attention 可能会过度关注最后这 1 维 |

**最终建议：**
虽然官方使用的是暴力的通道拼接（Concat）方案，但这是因为他们的模型非常浅且硬编码了维度（126+2=128）。
在我们当前的深度架构（DCSA 融合 + DSTFormer 时空注意力）中，**强烈建议采用“方案 A：特征加和”**。这不仅能达到与官方完全相同的“引入人物身份偏置”的目的，而且在 Transformer 架构中（如 ViT 的 class token 或 PE），相加（Add）是比拼接（Concat）更标准、更优雅的条件注入方式。