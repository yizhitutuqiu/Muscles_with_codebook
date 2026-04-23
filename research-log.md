# 研究日志

## 初始化
**日期:** 2026-04-17
**动作:** 初始化 autoresearch 项目 "MIA Temporal Tokenization Alignment"

- 审阅了 `custom/docs/` 目录下的 `PLAN.md`、`DECODER_AND_1E8_PARADOX.md` 和 `now_results.csv` 等文档和记录。
- 确认了核心目标 1：将 Stage 1 TCN codebook 的输出修改为帧对齐 token（`t * 63 + k`），使得 Stage 2 能够接收单帧离散输入，同时附带时序上下文。
- 确认了核心目标 2：优化算法，使得在不使用 cond (`stage2(no_cond)`) 时，模型的平均生成误差能降低到 9-10 之间（目前为 10.67，需超越 official_nocond 的 11.04）。
- 建立了 `H1` 假设：通过使用 `PLAN.md` 中的 "Plan A" (Frame-Aligned Temporal Tokenizer，利用 1x1 conv) 将能提供对齐且含上下文的 token 表示，这是降低 stage2(no_cond) 误差的关键基础。
- 用户指定了交互语言为中文，并且要求后续实验运行在一个名为 `mia_custom` 的 tmux 窗口中。

## 放弃 H1 与 Residual 模式，转向 H2
**日期:** 2026-04-17
**动作:** 根据用户反馈终止 H1 并修改研究方向

- **反馈与观察**：在尝试将 `emg_pred_mode` 改为 `residual`（即让 Stage 1 的预训练 Decoder 输出 Base EMG，再由 Stage 2 预测残差）的实验 `train_20260417_171117.log` 中，2000 step 时 RMSE 仍然高达 ~18，远不及之前使用 `full` 模式（从头预测 EMG）在 `train_20260415_112811.log` 中的 ~13.5。
- **用户确认**：
  1. `clip_5` (Unified clip level codebook with clip_len=5) 的表现明确优于 `frame_aligned` (逐帧扩展)。
  2. `full` 预测模式明确优于 `residual` 残差模式。
- **调整策略**：果断终止正在进行的残差预测实验。新的实验假设 `H2` 必须基于 `clip_5` 的 Stage 1 模型以及 `full` 模式的 Stage 2 模型。由于目标是在 `no_cond`（缺乏人物身份信息）下降低误差到 10 以下，我们必须从 Stage 2 的时序建模能力、融合能力或正则化手段（如 Dropout / 层数调整）入手进行算法级优化。

## 批量训练与评测机制重构 (H3 - H5)
**日期:** 2026-04-22
**动作:** 重构训练流程与自动化评测体系，修正评测脚本的关键隐藏 Bug

- **实验流重构**：编写了 `batch_train_stage2.py` 及其 shell 脚本，支持基于 YAML 文件的多 GPU 批量消融训练。
- **废弃 H4 转向 H5**：用户指出 H4 的“隐式偏置（Implicit Bias）”策略属于投机取巧。废弃 H4，真正从底层网络结构入手设计 **H5**：引入离散 Codebook Token 的时空位置编码、RoPE 旋转位置编码、局部时序卷积 (LTC) 和 SwiGLU 门控机制（DSTFormerV2）。
- **自动化评测集成**：修改 `Mia_style_eval.py` 使其能输出平均值，并编写 `batch_Mia_style_eval.py` 在训练后自动评测并输出汇总结果 `summary_processed.csv`。
- **评测 Bug 修复**：
  1. **设备不匹配 (Device Mismatch)**：修复了官方模型前向传播中硬编码 `torch.cuda.FloatTensor` 导致的在指定 GPU 设备上报错的问题。
  2. **路径丢失 (No eval cases found)**：修正了 `Mia_style_eval.py` 中 `id_exercises` 对应的 `ablation_dir` 绝对路径。
  3. **结果完全相同 (Identical Results Bug)**：发现虽然 `batch_Mia_style_eval.py` 生成了独立的评测指令，但由于未在生成的临时 YAML 中完整显式地覆盖 `methods.stage2.checkpoint` 字典结构，导致 `Mia_style_eval.py` 内部回退（Fallback）去读取了默认的 `eval.yaml`，从而所有的评测实际上都在测试同一个写死的老 checkpoint (`stage2_with_clip5temp_cond`)。修复：通过 `copy.deepcopy` 和完整显式地注入字典参数，成功切断了回退逻辑。