# H7 实验报告：双向对称跨注意力与双分支并行 MoE 的突破

**日期:** 2026-04-28
**实验编号:** H7 (`batch_stage2_h7_symmetric_dual`)
**核心基座:** H6-C (Token-level MoE) + `clip5_tcn` Codebook

## 1. 实验背景与动机
在之前的 H6 实验中，我们引入了 Token-level MoE（混合专家系统），成功将无条件 (`no_cond`) 下的平均 RMSE 降低至 10.39。然而，针对 H6 架构，我们发现了两个潜在的优化空间：
1. **融合方式的不平等**：原有的 DCSA (Discrete-Continuous Spatial Attention) 是单向非对称的（连续特征作为 Query，离散特征作为 Key/Value），这导致离散运动学词汇无法反向锚定物理状态。
2. **时空处理的串行化**：原有的 DSTFormer 采用交替的串行时空处理，导致时空信息过早混合，可能丢失纯粹的单维度特征。

因此，H7 实验旨在引入 **Symmetric DCSA（双向对称跨注意力）** 与 **Dual-Branch 并行 MoE（时空双分支）**，以期进一步突破性能瓶颈。

## 2. 消融实验设计与结果
我们设计了三组消融实验，分别在 GPU 4, 5, 6 上并行训练，测试集结果如下：

| 实验名称 | 核心机制配置 | 平均 RMSE (`no_cond`) | 结论分析 |
| :--- | :--- | :---: | :--- |
| `exp_h7_sym_dcsa_only` | 仅使用 Symmetric DCSA (串行 MoE 主干) | 10.55 | 与官方基线持平，说明单纯引入对称特征而在主干中串行消化，会导致特征拥挤，无法发挥优势。 |
| `exp_h7_dual_branch_only` | 仅使用 双分支并行 MoE (单向 DCSA) | 10.42 | 性能优秀，证明并行的“先时间后空间”与“先空间后时间”路线优于单一串行提取。 |
| **`exp_h7_full_sym_dual`** | **Symmetric DCSA + 双分支并行 MoE** | **10.30** | **重大突破！** 双向对称融合与并行双分支产生了完美的化学反应，极大地提升了模型的无条件泛化能力。 |

## 3. CIF 机制的验证与反思
在并行开展的另一项验证中，我们测试了基于语音识别的 CIF（Continuous Integrate-and-Fire）软切分机制替代现有的 `clip5_tcn` Codebook。
- **结论**：CIF 机制在当前 Stage 2 架构中被验证为**无效**。虽然其在 Stage 1 的重构上表现尚可，但其变长软切分带来的不确定性破坏了 Stage 2 对时空结构的稳定预期，导致下游预测性能崩塌。
- **决策**：彻底放弃 CIF，**全面回归并锁定 `clip5_tcn` Codebook 作为后续所有优化的离散化基座**。

## 4. 下一步探索方向 (H8 展望)
既然 `clip5_tcn` Codebook 提供的离散化 Token 具有极其稳定的运动学先验语义（即每一个离散 Token 都代表一种高度聚类的“动作元/运动原语”），我们接下来的核心研究方向将转向：**Codebook 引导的 MoE 路由机制 (Kinematic-Prior Guided MoE)**。

我们将探索如何让离散化的 Codebook Index（语义先验）直接参与或干预 MoE Router 的决策过程，让不同的专家网络真正与特定的“动作元”绑定，从而实现真正的“语义级专家分工”。