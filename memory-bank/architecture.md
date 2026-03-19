# Architecture

本清单重点覆盖源码、数据、配置、脚本和关键文档。
`.git/`、`__pycache__/`、时间戳日志等元数据或运行产物按类型汇总，不逐个展开。

## Root

- `.git/`: Git 元数据，不属于业务逻辑。
- `.gitignore`: 忽略规则。
- `.vscode/`: 本地编辑器设置，目前只看到 `settings.json`。
- `AGENTS.md`: 当前项目级 Codex 指令入口。
- `memory-bank/`: 当前项目的持久记忆文档目录。
- `README_experiments.md`: planner-only benchmark 的使用说明和实验口径。
- `REPORT.md`: 阶段性实验汇报，重点讲公平对比框架。
- `1-codingfix.md`: 早期模块规划笔记。
- `add_nag2g_after.md`: NAG2G 接入说明和示例命令。
- `project_analysis_report.md`: 某次 benchmark 优化结果分析。
- `list.txt`: 常用命令草稿。
- `environment.yml`: 主仓库 `retrogp` 环境配置。
- `nag2g.yml`: NAG2G 推理环境配置。
- `demo_utils.py`: demo 用的世界构造函数、数据加载小工具、默认 objective 构造。
- `model/`: 新加入的 encoder-decoder 实验目录，计划用于做单步逆合成条件生成。
- `run_repr_demo.py`: 表示层最小 demo，展示 `Program -> Route`。
- `run_feas_demo.py`: 可行性层 demo，展示 FG 预筛、动作掩码和可行执行。
- `run_obj_demo.py`: 多目标层 demo，展示 `RouteFitnessEvaluator`、lexicase、NSGA-II。
- `run_gp_search_demo.py`: 早期的单文件 GP 搜索 demo，偏演示性质，不是当前主线入口。
- `run_real_data_gp.py`: 当前主入口，按 YAML 目标配置运行真实数据上的 GP 搜索，可选挂接 NAG2G。
- `run_demod_until_solved.py`: 面向单个目标的“加大预算尝试求解”脚本。

## Runtime And Editor Dirs

- `logs/`: 运行日志、benchmark 输出、绘图输入目录。
- `logs/benchmarks/`: planner-only benchmark 输出目录。
- `logs/run_real_data_gp_output_*.txt`: 多次 `run_real_data_gp.py` 的标准输出快照。
- `__pycache__/` 与各包内 `__pycache__/`: Python 缓存目录，不参与逻辑。
- `.vscode/settings.json`: 本地 VS Code 设置，不影响 Python 行为。

## Data

- `data/`: 项目数据根目录。
- `data/get_building_block.py`: 从 ChemProjector 矩阵缓存提取命中的 building blocks，并可复制到 `data/building_block/`。
- `data/building_block/`: purchasable inventory 文件目录。
- `data/building_block/building_blocks_dataset.txt`: 默认 stock 文件，benchmark 和 GP 经常直接使用。
- `data/building_block/enamine_hb_matched_20251007.smi`: 一份匹配后的 building block 列表。
- `data/building_block/enamine_hb_matched_building_20251007.smi`: 另一份匹配后的 building block 列表，命名更接近 retrogp 使用场景。
- `data/building_block/test_building_blocks.smi`: 小型测试库存。
- `data/reaction_template/`: 反应模板目录。
- `data/reaction_template/hb.txt`: 当前默认 retro template 文件。
- `data/target molecular/`: 目标分子和命名 demo 配置目录。
- `data/target molecular/config.yaml`: DemoA-D 的目标、步长和 building block 数据集映射。
- `data/target molecular/chembl.txt`: 目标分子列表，看名称应是 ChEMBL 目标集。
- `data/target molecular/chembl_small.txt`: 更小的 ChEMBL 目标子集。
- `data/target molecular/enamine_smiles_1k.csv`: 一组 Enamine SMILES 数据。
- `data/target molecular/test_chembl.csv`: benchmark 或测试用 ChEMBL 目标集。
- `data/target molecular/test_synthesis.csv`: planner-only benchmark 默认目标文件。

## Experimental Encoder-Decoder

### `model/`

- `model/data/`: 新实验目录的数据集。
- `model/data/train.csv`: 训练集，列为 `class` 和空格切分后的 `reactants>reagents>production`。
- `model/data/eval.csv`: 验证集。
- `model/data/test.csv`: 测试集。
- `model/RETROSYNTHESIS_PLAN.md`: 针对当前 encoder、decoder 和 USPTO-full 风格数据集的单步逆合成实施计划书。
- `model/encoder/`: BERT 风格 encoder 目录。
- `model/encoder/local_bert.py`: 自定义双向 Transformer encoder，含 embedding、bidirectional self-attention、pooler、MLM head。
- `model/encoder/encoders.py`: encoder 包装层，负责按 spec 构建本地 BERT 或 HuggingFace encoder。
- `model/encoder/MolEncoder-SMILES-Drug-1.2B/`: encoder 权重、配置和词表目录。
- `model/encoder/MolEncoder-SMILES-Drug-1.2B/encoder.yaml`: encoder 结构配置，目前是 `24 x 32 x 2048`。
- `model/encoder/MolEncoder-SMILES-Drug-1.2B/checkpoint.pt`: encoder 预训练权重。
- `model/encoder/MolEncoder-SMILES-Drug-1.2B/vocab.txt`: encoder 词表。
- 当前源码里 `LocalBertEncoder` 构造 `SmilesTokenizer` 时没有显式传 `do_lower_case=False`，会把 `C/O` 等 token 降成小写，破坏化学语义。
- `model/decoder/`: GPT 风格 decoder 目录。
- `model/decoder/model.py`: 自回归 GPT 模型，使用 causal attention 和 RoPE，当前没有 cross-attention。
- `model/decoder/tokenizer.py`: decoder 专用 SMILES tokenizer，明确重写了 regex 分词逻辑。
- `model/decoder/loadmodel_example.py`: decoder 加载与生成示例。
- `model/decoder/vocabs/vocab.txt`: decoder 词表。
- `model/decoder/weights/SMILES-650M-3B-Epoch1.pt`: decoder 权重。
- 当前已删除 `model/decoder/model.py` 中孤立的 `utils.train_utils.Variable` 历史导入，decoder 代码现在可以直接导入。
- 当前已确认 encoder 和 decoder 的 `.pt` 权重文件都可以被 `torch.load` 正常读取，并且 decoder 权重与当前 GPT 结构严格匹配。

## Core Search Layer

### `gp_core/`

- `gp_core/__init__.py`: 包说明，表明这里是 modular GP retrosynthesis search helpers。
- `gp_core/config.py`: 默认路径、GP 超参数、objective 权重、SCScore 路径。
- `gp_core/budget.py`: planner-only 对比里最关键的 calls budget 计数器和 `BudgetExceeded`。
- `gp_core/data_loading.py`: inventory、targets、templates 的统一加载入口。
- `gp_core/templates.py`: 从文本模板文件构建 `ReactionTemplateRegistry`。
- `gp_core/program_ops.py`: GP 个体编码、随机初始化、交叉、变异。
- `gp_core/executor.py`: 组装 `FeasibleExecutor` 并把 `Program` 评估成 `{program, route, fitness}`。
- `gp_core/scoring.py`: 当前最核心的评分文件，包含 `StepScorer`、`route_state_from_route`、`GPFitnessEvaluator`。
- `gp_core/fitness.py`: 另一套偏兼容/桥接的 fitness helper，连接 `gp_retro_obj` 与 SCScore loader。
- `gp_core/search.py`: 主 GP 搜索循环，包含 parent ranking、进化过程、NSGA-II survivor 选择。
- `gp_core/metrics.py`: `MetricsHistory`，记录 proposals、repeats、top-k 均值与 AUC。
- `gp_core/baselines/`: planner-only baseline planners。

### `gp_core/baselines/`

- `gp_core/baselines/__init__.py`: `PlannerResult` 数据结构。
- `gp_core/baselines/greedy.py`: greedy planner，每步取 top-1 one-step 候选。
- `gp_core/baselines/beam.py`: beam search planner，按 molecule-set state 扩展。
- `gp_core/baselines/best_first.py`: best-first / A*-0 planner。

## Representation Layer

### `gp_retro_repr/`

- `gp_retro_repr/README.md`: 表示层模块说明。
- `gp_retro_repr/__init__.py`: 统一导出 `Molecule`、`Inventory`、`Program`、`Route` 等对象。
- `gp_retro_repr/molecules.py`: RDKit canonical SMILES 和 `Molecule` 包装。
- `gp_retro_repr/stock.py`: `Inventory`、`LeafCriteria`、`ChemHistorian`，定义 purchasable 与 leaf 行为。
- `gp_retro_repr/templates.py`: `ReactionTemplate` 与 `ReactionTemplateRegistry`，支持 rdchiral 和 RDKit 回退。
- `gp_retro_repr/step.py`: 顺序化 retrosynthesis step 结构。
- `gp_retro_repr/route.py`: `Route` 结构、连通性不变量、JSON 序列化。
- `gp_retro_repr/program.py`: `Instruction` 集合、`Program`、解释器执行逻辑。
- `gp_retro_repr/validation.py`: molecule / reaction / route 三层检查。
- `gp_retro_repr/registry.py`: 从 JSONL 加载模板注册表的小工具。

## Feasibility Layer

### `gp_retro_feas/`

- `gp_retro_feas/README.md`: 可行性层说明。
- `gp_retro_feas/__init__.py`: 统一导出 FG 检测、typing、mask、executor 等接口。
- `gp_retro_feas/fg_patterns.py`: 官能团 SMARTS 库和检测函数。
- `gp_retro_feas/typing.py`: 模板家族和 FG 需求的轻量 typing 系统。
- `gp_retro_feas/preselect.py`: `TemplatePreselector`，先用 FG/type 做快速候选裁剪。
- `gp_retro_feas/engine.py`: `FeasibilityEngine`，负责模板应用、清洗输出、库存门控、选 reactant set。
- `gp_retro_feas/mask.py`: `ActionMaskBuilder`，把候选模板过滤成“可行动作集合”。
- `gp_retro_feas/executor.py`: `FeasibleExecutor`，执行 `Program`，支持模板修复和 one-step model 动作。

## Objective Layer

### `gp_retro_obj/`

- `gp_retro_obj/README.md`: 多目标层说明。
- `gp_retro_obj/__init__.py`: 对外导出 objective、fitness、selection 接口。
- `gp_retro_obj/objectives.py`: `ObjectiveSpec` 和 objective 类型定义。
- `gp_retro_obj/fitness.py`: `RouteFitnessEvaluator`、`FitnessResult`、scalarization、属性 oracle 辅助。
- `gp_retro_obj/selectors.py`: `epsilon_lexicase_select` 和 `nsga2_survivor_selection`。
- `gp_retro_obj/oracles.py`: QED oracle 和简单注册表。

## One-Step Adapter Layer

### `gp_retro_nn/`

- `gp_retro_nn/__init__.py`: 对外导出 one-step 协议和 NAG2G 适配器。
- `gp_retro_nn/one_step.py`: `OneStepPrediction` 和 `OneStepRetrosynthesisModel` 协议。
- `gp_retro_nn/nag2g_subprocess.py`: 用持久子进程方式包装 NAG2G 推理服务，并带本地缓存。
- `gp_retro_nn/nag2g_server.py`: 被子进程调用的 JSONL 服务端，真正初始化 NAG2G 模型并执行推理。

## SCScore Helpers

### `multi-score/`

- `multi-score/README.md`: SCScore partial reward patch 说明。
- `multi-score/__init__.py`: 导出 SCScore 接口。
- `multi-score/scscore_reward.py`: SCScore 模型包装、磁盘缓存、partial reward 公式。

### `scscore/`

- `scscore/`: vendored upstream SCScore 项目。
- `scscore/data/`: 上游数据目录。
- `scscore/models/`: 预训练模型目录，本仓库默认 SCScore 路径也指向这里。
- `scscore/notebooks/`: 上游 notebook。
- `scscore/scripts/`: 上游工具脚本。
- `scscore/scscore/`: 上游 Python 包源码。
- `scscore/utils/`: 上游辅助脚本。
- `scscore/README.md`: 上游说明文档。
- `scscore/LICENSE`: 上游许可证。

## Experiment Scripts

### `scripts/`

- `scripts/benchmark_planners.py`: planner-only benchmark 主脚本，统一调用四种 planner 并输出 CSV。
- `scripts/analyze_benchmarks.py`: 读取 `per_target.csv`，输出 summary 和 run 对比报告。
- `scripts/scoring_smoke_test.py`: 当前最关键的轻量 smoke test，覆盖 StepScorer、GPFitnessEvaluator、leaf 逻辑和 executor 行为。
- `scripts/inspect_nag2g_one_step.py`: 直接查看 NAG2G raw 输出和 StepScorer 重排结果。
- `scripts/plot_gp_gen.py`: 把 GP 每代指标 CSV 画成图。
- `scripts/fetch_uspto_templates.py`: 从 HuggingFace 数据集拉取反应 SMILES 并导出 retro 风格模板列表。

## Settings Snapshot

### `setting/`

- `setting/save_vars.py`: 保存当前 GP / executor 配置快照到 JSON。

## Third-Party Bundles

### `ASKCOS/`

- `ASKCOS/askcos/`: upstream ASKCOS Python 包。
- `ASKCOS/deploy/`: 部署相关文件。
- `ASKCOS/docs/`: upstream 文档。
- `ASKCOS/makeit/`: ASKCOS retrosynthesis 逻辑目录，本仓库 leaf 行为曾参考这里。
- `ASKCOS/README.md`: upstream 说明。
- `ASKCOS/requirements.txt`: upstream 依赖。
- `ASKCOS/Dockerfile`: upstream 容器定义。
- `ASKCOS/LICENSE`: upstream 许可证。

### `ChemProjector-main/`

- `ChemProjector-main/assets/`: upstream 静态资源。
- `ChemProjector-main/bin/`: 可执行辅助脚本。
- `ChemProjector-main/chemprojector/`: upstream 主包源码。
- `ChemProjector-main/chemprojector.egg-info/`: 打包元数据。
- `ChemProjector-main/configs/`: 配置目录。
- `ChemProjector-main/data/`: 上游数据目录，也与 building block 预处理有关。
- `ChemProjector-main/scripts/`: upstream 脚本。
- `ChemProjector-main/third_party/`: 上游依赖代码。
- `ChemProjector-main/README.md`: upstream 说明。
- `ChemProjector-main/env.yml`: upstream 环境配置。
- `ChemProjector-main/pyproject.toml`: 打包配置。
- `ChemProjector-main/setup.py`: 打包脚本。
- `ChemProjector-main/train.py`: 训练入口。
- `ChemProjector-main/sample.py`: 采样示例。
- `ChemProjector-main/unarchive_wizard.py`: 数据解包工具。
- `ChemProjector-main/LICENSE`: upstream 许可证。

### `NAG2G-main/`

- `NAG2G-main/NAG2G/`: upstream 主包源码。
- `NAG2G-main/Uni-Core-main/`: upstream 依赖副本。
- `NAG2G-main/unimol_plus/`: upstream 相关包。
- `NAG2G-main/data_preprocess/`: 数据预处理脚本。
- `NAG2G-main/outputs/`: 训练或推理输出目录。
- `NAG2G-main/USPTO50K_brief_20230227/`: 数据目录。
- `NAG2G-main/NAG2G_unimolplus_uspto_50k_20230513-222355/`: checkpoint 目录。
- `NAG2G-main/README.md`: upstream 说明。
- `NAG2G-main/2024-NAG2G.md`: 项目说明或记录文档。
- `NAG2G-main/train.sh`: 训练脚本。
- `NAG2G-main/valid.sh`: 验证脚本。
- `NAG2G-main/run_train_classknow_smoke.sh`: smoke training 脚本。
- `NAG2G-main/check_lmdb_class_field.py`: 数据检查脚本。
- `NAG2G-main/normalize_smi_tokens_for_g2g_cal.py`: SMILES token 归一化脚本。
- `NAG2G-main/strip_class_prefix_from_smi.py`: token 清洗脚本。
- `NAG2G-main/tg-run.txt`: 运行记录或命令草稿。
- `NAG2G-main/LICENSE`: upstream 许可证。

## Historical Notes

- `gp_core/scoring.py` 中的 `GPFitnessEvaluator` 是当前 planner-only 与 one-step 排序主线。
- `gp_core/fitness.py` 仍然有用，但更像旧接口或桥接层，不要和 `gp_core/scoring.py` 的职责混淆。
- `run_gp_search_demo.py` 展示了项目早期从单文件脚本走向模块化的痕迹，排查历史行为时可以参考，但不要默认把它当成当前生产入口。
