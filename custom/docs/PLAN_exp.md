## 目标

在不改 Stage1/codebook（仍然用 In-Distribution 训练好的 codebook、并在 Stage2 中冻结）的前提下，复现官方 Muscles in Action 的 **Out-of-Distribution（OOD）** Pose→EMG 实验口径，并用相同的 OOD 数据划分训练/评测我们的 Stage2，与官方方法（cond on/off 开关）及 Retrieval baseline 对表。

约束：
- 不改 Stage1/codebook 训练。
- 仅在 [`train_stage2_pose2emg.py`](file:///data/litengmo/HSMR/mia_custom/custom/stage2/train/train_stage2_pose2emg.py) 增加 OOD 适配与自动化跑实验能力（实现时）。
- 当前只做 Pose→EMG（不做 EMG→Pose）。

## 官方 OOD 实验到底是什么（需要对齐的“口径”）

官方在 `golf_third_party/musclesinaction/musclesinaction/ablation/` 下提供了 OOD 的 train/val filelist，且其命名与推理脚本一致（见官方 `inference_commands` / `inference_scripts`）。

我们需要对齐的 OOD 主要有两类：

### 1) OOD Exercises（跨动作泛化 / per-exercise）

目录：
- `/data/litengmo/HSMR/golf_third_party/musclesinaction/musclesinaction/ablation/generalizationexercises/`

文件模式：
- `train_<Exercise>.txt`
- `val_<Exercise>.txt`

含义（从 filelist 内容可验证）：
- `val_<Exercise>.txt`：验证集只包含该动作 `<Exercise>`（且来自某个 Subject 的 val 段，例如 `val_Running.txt` 里是 Subject5/Running）。
- `train_<Exercise>.txt`：训练集由“其他动作/其他 subject 的 train 段”拼成，用于模拟“对该动作的 OOD 泛化”。

### 2) OOD People（跨人泛化 / per-subject）

目录：
- `/data/litengmo/HSMR/golf_third_party/musclesinaction/musclesinaction/ablation/generalizationpeople/`

文件模式：
- `train_Subject<k>.txt`
- `val_Subject<k>.txt`

含义：
- `val_Subject<k>.txt`：验证集只包含 Subject<k> 的 val 段（可能覆盖多个动作）。
- `train_Subject<k>.txt`：训练集由其他 subject 的 train 段拼成（通常不包含 Subject<k>），用于跨人泛化。

备注（关于 Conditioning）：
- 官方没有提供“nocond 专属 checkpoint”的必要性；其模型接口始终接收 `condval`，但通过 dataloader 的 `cond=True/False` 来控制 condval 的生成方式（cond=True 按 Subject 映射不同 condval；cond=False 固定 condval 常数）。我们在对表中将“official_cond/official_nocond”理解为同一个官方 checkpoint 在评测时分别用 cond on/off。

## 我们的 OOD 实验设计（跑法与输出组织）

### 核心思路

对每个 OOD case（一个动作或一个 subject），我们都训练一个 Stage2 模型（Stage1/codebook 固定为同一个 ID codebook），然后在对应的 val filelist 上评测并输出官方口径指标。

为什么需要“每个 case 训练一个 Stage2”：
- 官方 OOD 是“重新训练模型以适应某个 OOD 任务”，并非只评测一个 ID 训练的模型在 OOD 上的 zero-shot 泛化。
- 我们要与官方对齐，必须让 Stage2 的训练数据也切换成相同 OOD train split。

### Stage2 的初始化策略（建议同时支持两种）

为了既对齐官方、又方便探索，我们建议 Stage2 OOD 训练支持两种初始化方式：

1) **from_scratch**：Stage2 随机初始化（保持与官方“每个 OOD case 单独训练”更接近）。
2) **warm_start_from_ID**（推荐默认）：先加载一个 ID 训练好的 Stage2 checkpoint 作为初始化，再在 OOD train split 上 finetune。
   - 直觉：Stage1 tokens 与融合骨干学到的“EMG 映射结构”可迁移，OOD 训练只需适配分布偏移。
   - 工程收益：训练更稳定、更快，且更易跑完大量 case。

## 需要在 train_stage2_pose2emg.py 增加的“OOD 适配”能力（实现设计）

### 1) 配置层：新增 ood 配置块 + 统一 run 生成规则

在 Stage2 的训练 config（yaml）里新增（或扩展）字段，例如：
- `ood.enabled: bool`
- `ood.protocol: exercises | people`
- `ood.ablation_dir: <path>`
- `ood.targets: null | [Running, FrontKick, ...] | [Subject0, Subject1, ...]`
  - null 表示自动扫描 ablation_dir 下所有满足模式的 val filelist 作为 targets
- `ood.train_pattern / ood.val_pattern`：
  - exercises：`train_{target}.txt` / `val_{target}.txt`
  - people：`train_{target}.txt` / `val_{target}.txt`（target 为 `Subject5` 这类）
- `ood.init: scratch | warm_start`
- `ood.init_stage2_checkpoint: <path>`（warm_start 时必填）
- `ood.out_root: <dir>`：所有 OOD runs 的输出根目录（每个 target 一个子目录）
- `ood.max_steps_override / ood.lr_override / ood.batch_size_override`：允许对 OOD 训练单独覆写超参（可选）

输出目录建议结构（便于后续评测脚本批量解析）：
- `<ood.out_root>/<protocol>/<target>/`
  - `train.log`
  - `last.pt`, `best.pt`
  - `config_snapshot.yaml`（当次 run 的最终配置展开，便于复现实验）
  - `train_data_paths.txt`（你现有脚本已有类似逻辑，建议每个 target 独立保存）

### 2) 运行层：训练脚本支持 “单次训练” 与 “批量 OOD sweep”

在训练入口增加两种工作模式：

- **标准模式（现有）**：只按 `data.train_filelist` + `data.val_filelist` 训练一次。
- **OOD sweep 模式（新增）**：当 `ood.enabled=True` 时：
  1. 解析/发现 targets 列表
  2. 对每个 target：
     - 将 train/val filelist 切换到 OOD 对应的 `train_*.txt` / `val_*.txt`
     - 训练一次 Stage2（可选 warm_start）
     - 保存 best/last、日志等到 target 子目录

关键设计点：
- 支持 `--only_target <name>`：用于只跑某个动作/subject（调试非常重要）。
- 支持 `--dry_run`：只打印将要跑的 targets 与路径，不真正训练。
- 支持 `--resume_target`：某个 target 已有 last.pt 时是否续训，避免中途断掉重跑。

### 3) 数据层：与官方 filelist 兼容（无需新数据格式）

官方 filelist 每行形如：
- `MIADatasetOfficial/train/Subject5/LegBack/1105`

现有 `MyMuscleDataset` 直接读取这种路径格式即可；因此 OOD 适配的重点是“切换 filelist 文件”，而不是改 dataloader。

同时建议在 OOD sweep 中加入一致性检查（仅打印/断言）：
- train/val filelist 文件存在且非空
- 随机抽查若干行，确认路径前缀为 `MIADatasetOfficial/`
- `step==30`（官方所有这些实验都在 30 帧窗口下）

### 4) Conditioning 的处理（我们的 Stage2 不用 cond，但要保证口径一致）

Stage2 当前 forward 只吃 joints3d，不吃 condval，因此训练时 `cond=True/False` 对 Stage2 实际学习无影响。

但为了和官方评测“cond on/off”对应，我们建议：
- Stage2 训练时固定使用 `cond=False`（更接近“无条件输入”设定），或固定 `cond=True`（更接近官方默认）；二者对 Stage2 不影响，但能保证输出目录里记录清楚。
- 在 config snapshot 里记录当次 run 使用的 `data.cond` 值，避免后续对齐时混淆。

### 5) 指标与 best checkpoint 选择（保持官方口径）

训练过程中的验证集 best 选择建议继续使用 “raw 空间 RMSE”：
- 若训练使用 `emg_normalize_target=true`（full 模式），验证时需要像现有训练脚本那样对 pred 做反归一化后再算 RMSE（当前脚本已实现类似逻辑）。
- best checkpoint 以 `val_rmse` 最小为准（与你现有 Stage2 train 脚本一致）。

### 6) 计算成本与稳定性建议（OOD 会更难收敛）

OOD sweep 可能需要跑 10（people）+ 15（exercises）个 target：
- 建议 warm_start 默认开启，并允许对 OOD 覆写较小 lr（例如 ID lr 的 0.1~0.5 倍）、较短 max_steps（先跑通 pipeline）
- 每个 target 训练日志独立保存，方便定位某个 target 是否发散

## 最终对表如何组织（为下一步评测脚本扩展铺路）

完成 OOD 训练后，我们会得到每个 target 的 Stage2 checkpoint。后续评测阶段（下一步再做代码）可以沿用我们现有 `Mia_style_eval.py` 的思路，扩展一个 `ood_exercises` / `ood_people` protocol：
- 对每个 target：
  - 用相同 val filelist 分别评测：
    - 我们 Stage2（对应 target 的 best.pt）
    - 官方模型 cond=True
    - 官方模型 cond=False
    - Retrieval baseline（用同一个 OOD train filelist 建库，再在 val filelist 评测）

输出应保持与 ID 相同的 `summary.csv` 结构，便于用已有的 `process_mia_style_summary.py` 做可视化/汇总。

## 实施步骤（只设计，不改代码）

1. 确认 OOD filelists 全部存在：
   - generalizationexercises: `train_*.txt` 与 `val_*.txt`
   - generalizationpeople: `train_Subject*.txt` 与 `val_Subject*.txt`
2. 设计 Stage2 OOD 训练配置 schema（上述 `ood.*`）
3. 在 `train_stage2_pose2emg.py` 增加 OOD sweep 模式：
   - 自动 discover targets
   - per-target 覆写 train/val filelist
   - per-target 输出目录与断点续训
   - warm_start 支持
4. 先用 1 个 target 做 smoke test：
   - exercises: `Running`
   - people: `Subject5`
5. 跑完整 sweep（或挑选部分 targets）
6. 再扩展评测脚本将 OOD protocols 纳入统一对表（后续步骤）
