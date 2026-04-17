# Stage2 网络全流程与数据流（基于 stage2_pose2emg.yaml）

本文档基于**当前** `custom/stage2/configs/stage2_pose2emg.yaml`，逐步说明从输入到输出的**数据形状与流动**，便于做详细分析。

**当前 YAML 关键配置**：`cont_encoder_type: joint_25`，`fusion_type: dcsa_asymmetric`，`temporal_type: dstformer`，`emg_head_type: flatten`，`emg_pred_mode: full`；**时空位置编码 (PE)** 在 z_cont 送入 DCSA 前注入；训练时 **emg_normalize_target: true**（目标用 Stage1 Standardizer 归一化，指标在 raw 空间）。

---

## 符号约定

| 符号 | 含义 | 当前配置取值 |
|------|------|--------------|
| B    | batch size | 16（data.batch_size） |
| T    | 时间帧数   | 30（data.step） |
| J    | 关节点数   | 25（与 joints3d 一致） |
| N    | 离散 token 数（Stage1 codebook） | 63（model.token_count） |
| Nf   | 融合后 token 数（= 连续分支 token 数） | 25（joint_25 + dcsa_asymmetric 时） |
| C / dim | 特征维度 | 256（model.dim） |

---

## 一、总览：从 joints3d 到 emg_pred

```
输入: joints3d (B, T, 25, 3)
        │
        ├──► 离散分支 (Stage1 冻结): 每帧 75 维 → normalize → encoder → VQ → z_disc (B*T, 63, 256)
        │
        └──► 连续分支 (可训练):      每帧 (25, 3) → MLP(3→128→256)+LN 逐关节 → z_cont (B*T, 25, 256)
        │
        ▼
    view → z_cont_bt (B,T,25,256)，注入 PE: z_cont_bt += spatial_pe + temporal_pe[:,:t]
        │
        ▼
    融合 (dcsa_asymmetric): Q=z_cont_bt(25), K/V=z_disc_bt(63) → z_fused (B, T, 25, 256)
        │
        ▼
    时序 (dstformer):       z_fused → z_out (B, T, 25, 256)，保留 25 不做 mean
        │
        ▼
    EMG 头 (flatten):       z_out (B, T, 25, 256) → reshape (B,T,6400) → LN→Linear(6400,256)→GELU→Linear(256,8) → emg_pred (B, T, 8)
        │
        ▼
    输出策略 (full):        emg_pred = emg_out（无 Stage1 保底）
```

**训练时**：若 `emg_normalize_target: true` 且 Stage1 有 EMG standardizer，则 target 先归一化再算 loss；日志与 val 的 MAE/MSE 在 **raw 空间**（pred 反归一化后与 raw gt 比较）。梯度裁剪 `grad_clip_norm: 1.0` 在 backward 后、step 前执行。

---

## 二、逐步数据流（当前配置：joint_25 + PE + dcsa_asymmetric + dstformer + flatten + full）

### Step 0：输入与展平

- **输入**：`joints3d`，形状 `(B, T, 25, 3)`，已按需做 root-center。
- **展平**：  
  `x_flat = joints3d.reshape(b * t, 75)`  
  得到 `(B*T, 75)`，即每帧 75 维（25 关节 × 3 坐标）。后续连续分支在 joint_25 模式下会再 view 成 `(B*T, 25, 3)` 使用。

```
joints3d: (B, T, 25, 3)  →  x_flat: (B*T, 75)
```

---

### Step 1a：离散分支（Stage1 冻结，no_grad）

对 `x_flat (B*T, 75)` 逐帧独立处理。

1. **归一化**  
   `x_n = stage1.joints3d.normalize(x_flat, update=False)` → `(B*T, 75)`。

2. **编码**  
   `z_e = stage1.joints3d.encoder(x_n)` → 输出 `(B*T, N*C)`，`view(b, n, d)` → `z_e (B*T, 63, 256)`。

3. **VQ 量化**  
   `z_e` 展平送入 `stage1.vq`，得到 `z_q` 再 `view(b, n, d)` → **z_disc (B*T, 63, 256)**，以及 `idx (B*T*63,)`。

**小结**：  
`x_flat (B*T, 75)` → 离散分支 → **z_disc (B*T, 63, 256)**，`idx (B*T*63,)`。

---

### Step 1b：连续分支（可训练，当前：joint_25）

- **当前配置**：`cont_encoder_type: joint_25`，`cont_joint_hidden_dim: 128`。  
  - `x_joints = x_flat.view(b * t, 25, 3)` → 每帧 25 个关节、每关节 3 维，形状 **(B*T, 25, 3)**。  
  - `z_cont = self.cont_encoder(x_joints)`。`cont_encoder` 为**小 MLP**（对最后一维 3 逐关节作用）：  
    - `Linear(3, 128)` → `GELU()` → `Linear(128, 256)` → `LayerNorm(256)`  
    - 输出 **z_cont (B*T, 25, 256)**：保留 25 个关节的物理拓扑；小 MLP 避免弱 Query 冷启动。

**小结**：  
`x_flat (B*T, 75)` → view `(B*T, 25, 3)` → MLP(3→128→256)+LayerNorm → **z_cont (B*T, 25, 256)**。

---

### Step 2：整理为 (B,T,?,C)、注入 PE、送入融合

- `z_disc_bt = z_disc.view(b, t, 63, 256)` → **(B, T, 63, 256)**  
- `z_cont_bt = z_cont.view(b, t, 25, 256)` → **(B, T, 25, 256)**  

- **时空位置编码（仅 joint_25 时存在）**：  
  `z_cont_bt = z_cont_bt + self.spatial_pe + self.temporal_pe[:, :t, :, :]`  
  - `spatial_pe`: (1, 1, 25, 256)，为 25 个关节赋予空间身份  
  - `temporal_pe`: (1, max_seq_len, 1, 256)，取前 t 帧 → (1, t, 1, 256)，为时间维赋予顺序  
  - 注入后 **z_cont_bt** 仍为 **(B, T, 25, 256)**，再送入融合。

- **融合**：`z_fused = fusion(z_cont_bt, z_disc_bt)`，输出形状**跟随 Query（连续侧）**。

#### 当前配置：dcsa_asymmetric（非对称 DCSA）

- **Query**：`z_cont_bt (B, T, 25, 256)`（已带 PE）  
- **Key / Value**：`z_disc_bt (B, T, 63, 256)`  
- 实现：将 B、T 展平，`query (B*T, 25, 256)`，`key/value (B*T, 63, 256)`，做 Cross-Attention；输出 (B*T, 25, 256)，reshape 为 (B, T, 25, 256)，并与 `cont` 残差加后 LayerNorm。  
- 公式：`z_fused = LayerNorm(cont + Dropout(Attn(Q=cont, K=disc, V=disc)))`，输出 **(B, T, 25, 256)**。

**小结**：  
`z_cont_bt (B,T,25,C)`（+ PE）、`z_disc_bt (B,T,63,C)` → dcsa_asymmetric → **z_fused (B, T, 25, 256)**。

---

### Step 3：融合后时序模块（当前：DSTFormer）

- **输入**：`z_fused (B, T, 25, 256)`。  
- **DSTFormer**：4 层 Block，每层先对 25 个 token 做 spatial attention，再对 T 帧做 temporal attention，输出形状不变。  
- **不做 mean**：直接输出 **z_out (B, T, 25, 256)**。

**小结**：  
`z_fused (B,T,25,256)` → DSTFormer → **z_out (B, T, 25, 256)**。

---

### Step 4：EMG 头（当前：Flatten，保留 25 关节拓扑）

- **输入**：`z_out (B, T, 25, 256)`。  
- **当前配置**：FlattenEMGHead（保留关节维度信息，不做 mean 池化，避免局部肌肉混叠）。  
  - `flat = z_out.reshape(b, t, 25*256)` → **(B, T, 6400)**  
  - `self.net(flat)`：LayerNorm(6400) → Linear(6400, 256) → GELU → Linear(256, 8) → **(B, T, 8)**  
  - 记为 **emg_out**，在 full 模式下即 **emg_pred**。

**小结**：  
`z_out (B,T,25,256)` → reshape (B,T,6400) → LayerNorm→Linear(6400,256)→GELU→Linear(256,8) → **emg_pred (B, T, 8)**。

---

### Step 5：输出策略（当前：full）

- **emg_pred_mode: full**  
  - **emg_pred = emg_out**，无 Stage1 保底，直接由融合后的 25-token 特征回归 8 维 EMG。  
  - 训练时若 `emg_normalize_target: true`：模型输出在**归一化空间**（与 target 一致）；评估与日志中需用 Stage1.emg.standardizer 反归一化到 raw 空间再算 MAE/RMSE。

**小结**：  
**emg_pred (B, T, 8) = emg_out**。

---

## 三、形状汇总表（当前配置）

| 阶段           | 变量名      | 形状             | 说明 |
|----------------|-------------|------------------|------|
| 输入           | joints3d    | (B, T, 25, 3)    | 根心化后的 3D 关节 |
| 展平           | x_flat      | (B*T, 75)        | 每帧 75 维 |
| 离散分支       | z_disc      | (B*T, 63, 256)   | Stage1 encoder + VQ |
| 连续分支       | z_cont      | (B*T, 25, 256)   | joint_25: MLP(3→128→256)+LN 逐关节 |
| 融合前         | z_disc_bt   | (B, T, 63, 256)  | 离散 63 Token |
| 融合前         | z_cont_bt   | (B, T, 25, 256)  | 连续 25 Token（+ PE 后仍同形） |
| 融合后         | z_fused     | (B, T, 25, 256)  | dcsa_asymmetric，输出随 Query=25 |
| 时序后         | z_out       | (B, T, 25, 256)  | DSTFormer，保留 25 |
| EMG 头输出     | emg_out     | (B, T, 8)        | Flatten: (B,T,6400)→LN→Linear→GELU→Linear(256,8) |
| 最终输出       | emg_pred    | (B, T, 8)        | full: emg_pred = emg_out |

当前：B=16，T=30，J=25，N=63，Nf=25，C=256。

---

## 四、训练时的损失与监督（当前配置）

- **标签**：dataloader 的 `emg_values` 为 `(B, 8, T)`，permute 为 **gt_raw = (B, T, 8)**（raw 空间）。  
- **若 emg_normalize_target: true 且 Stage1.emg.standardizer 存在**：  
  - `gt_norm = (gt_raw - mean) / std`，**loss = smooth_l1_loss(pred, gt_norm)**（归一化空间）。  
  - 日志与 val：`pred_raw = pred * std + mean`，**loss_mae / loss_mse / val_smooth_l1** 均在 **raw 空间**（pred_raw vs gt_raw）。  
  - **梯度裁剪**：`loss.backward()` 后、`optimizer.step()` 前执行 `clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)`，默认 1.0。  
- **反向**：仅更新连续分支、PE、融合、时序、EMG 头；Stage1 全部冻结。

---

## 五、逐步数据流详解（用于详细分析）

以下按**代码执行顺序**写出每一步的输入/输出形状与含义。

| 步骤 | 位置 | 输入形状 | 操作 / 子模块 | 输出形状 | 说明 |
|------|------|----------|----------------|----------|------|
| 0 | forward 入口 | (B, T, 25, 3) | `joints3d.reshape(b*t, 75)` | (B*T, 75) | 根心化后的 3D 关节，展平为每帧 75 维 |
| 1a-1 | Stage1 离散 | (B*T, 75) | `stage1.joints3d.normalize(..., update=False)` | (B*T, 75) | 归一化，不更新 running 统计量 |
| 1a-2 | Stage1 离散 | (B*T, 75) | `stage1.joints3d.encoder(x_n)` | (B*T, N*C) | N=63, C=256 → view 成 (B*T, 63, 256) |
| 1a-3 | Stage1 离散 | (B*T, 63, 256) | `stage1.vq`，取 z_q | (B*T, 63, 256) | VQ 量化，得到 z_disc 与 idx |
| 1b-1 | 连续分支 | (B*T, 75) | `x_flat.view(b*t, 25, 3)` | (B*T, 25, 3) | 每帧 25 关节×3 坐标 |
| 1b-2 | 连续分支 | (B*T, 25, 3) | `cont_encoder(x_joints)`：Linear(3,128)→GELU→Linear(128,256)→LayerNorm | (B*T, 25, 256) | 逐关节 MLP，得到 25 个 Query token |
| 2-1 | 融合前 | (B*T, 63, 256), (B*T, 25, 256) | `z_disc.view(b,t,63,256)`, `z_cont.view(b,t,25,256)` | (B,T,63,256), (B,T,25,256) | 恢复 B,T 维度 |
| 2-2 | 位置编码 | (B, T, 25, 256) | `z_cont_bt += spatial_pe + temporal_pe[:, :t, :, :]` | (B, T, 25, 256) | 空间 PE (1,1,25,C) + 时间 PE (1,t,1,C) |
| 2-3 | 融合 | (B,T,25,256), (B,T,63,256) | `AsymmetricDCSA`: Q=cont, K/V=disc，MHA 后残差+LayerNorm | (B, T, 25, 256) | 25 个 Query 对 63 个 K/V 做 Cross-Attn |
| 3-1 | 时序 | (B, T, 25, 256) | `DSTFormer`: 4×Block（spatial Attn + temporal Attn），out_norm | (B, T, 25, 256) | 时空注意力，形状不变 |
| 4-1 | EMG 头 | (B, T, 25, 256) | `emg_head(z_out)`: reshape (B, T, 25*256) | (B, T, 6400) | 每帧 25×256 展平，保留 batch/time |
| 4-2 | EMG 头 | (B, T, 6400) | `LayerNorm(6400)` → `Linear(6400,256)` → `GELU` → `Linear(256,8)` | (B, T, 8) | **emg_pred**（full 模式即最终输出） |
| 5 | 输出 | (B, T, 8) | full 模式：emg_pred = emg_out | (B, T, 8) | 无 Stage1 保底；训练用归一化 target 时推理输出为归一化空间，评估需反归一化 |

---

## 六、小结（用于分析时可抓的点）

1. **连续分支 joint_25**：75 维保留为 (25, 3)，逐关节小 MLP(3→128→256)+LayerNorm，得到 25 个物理关节点 Token；`cont_joint_hidden_dim: 128` 缓解弱 Query 冷启动。  
2. **时空位置编码**：在 z_cont_bt 送入 DCSA 前加上 spatial_pe（25 维）与 temporal_pe（T 维），使 25 个 Token 具备空间身份与时间顺序。  
3. **非对称 DCSA**：Query=25 关节（带 PE）、K/V=63 离散积木，输出 25 Token；25 关节各自向 63 个先验检索，再残差+Norm。  
4. **Flatten EMG 头**：不在关节点维做 mean，保留 (B,T,6400) 后 LN→Linear(6400,256)→GELU→Linear(256,8)，保留“哪个关节对应哪块肌肉”的拓扑，避免 Spatial Pooling 混叠。  
5. **full 模式**：emg_pred = emg_out，无 Stage1 保底；配合 **emg_normalize_target** 在归一化空间监督，梯度裁剪驯服深层 Transformer，指标与 val 在 raw 空间（反归一化后）统一。

以上即为基于当前 `stage2_pose2emg.yaml` 的网络全流程与数据流说明，可直接用于详细分析或画图。
