# AGENTS.md

本文件是 `/data1/ytg/retrogp` 的项目级 Codex 记忆入口。

进入本仓库后，先读本文件，再按顺序读取 `memory-bank/`。如果任务只涉及一个局部模块，也先过一遍 `project-context.md` 和 `architecture.md`，避免改错层级。

## First Read

1. `memory-bank/project-context.md`
2. `memory-bank/architecture.md`
3. `memory-bank/tech-stack.md`
4. `memory-bank/implementation-plan.md`
5. `memory-bank/progress.md`

## Repo Boundaries

- 核心可维护代码在 `gp_core/`、`gp_retro_repr/`、`gp_retro_feas/`、`gp_retro_obj/`、`gp_retro_nn/`、`model/`、`scripts/`、`setting/`、`data/`、`run_*.py`、`demo_utils.py`。
- 主要运行入口是 `run_real_data_gp.py` 和 `scripts/benchmark_planners.py`。
- `ASKCOS/`、`ChemProjector-main/`、`NAG2G-main/`、`scscore/` 是内嵌第三方项目。默认不要直接改这些目录，除非用户明确要求，或者任务本身就是在这些子项目里开发。
- `logs/`、`__pycache__/`、benchmark 产物、时间戳输出文件都视为运行时产物，默认不编辑。

## Task Routing

- 表示层、路线、库存、leaf 判定、模板对象：先看 `gp_retro_repr/`。
- 可行性筛选、动作掩码、执行修复：先看 `gp_retro_feas/`。
- 多目标、适应度、选择器：先看 `gp_retro_obj/`，再看 `gp_core/fitness.py` 和 `gp_core/scoring.py` 的调用关系。
- GP 搜索、planner-only 公平对比、calls budget：先看 `gp_core/` 和 `scripts/benchmark_planners.py`。
- NAG2G one-step 接入：先看 `gp_retro_nn/`、`run_real_data_gp.py`、`scripts/inspect_nag2g_one_step.py`。
- 新上传的 encoder/decoder 与后续单步模型对齐模块：先看 `model/encoder/`、`model/decoder/`、`model/data/`。
- 数据、模板、目标配置：先看 `data/` 和 `gp_core/data_loading.py`。

## Working Rules

- 优先改本仓库自己的适配层和包装层，不要优先改 vendored upstream 代码。
- 保持 planner-only benchmark 的公平性前提不被破坏：
  same one-step model, same stock/leaf criterion, budget measured in actual one-step calls.
- 如果修改 one-step 候选排序或路线评分，必须联动检查 `gp_core/scoring.py` 和 `scripts/scoring_smoke_test.py`。
- 如果修改 leaf/stop 行为，必须联动检查 `gp_retro_repr/stock.py`、`run_real_data_gp.py`、`scripts/benchmark_planners.py`。
- 如果修改模板或库存加载，必须联动检查 `gp_core/data_loading.py`、`data/` 中的文件命名和调用脚本。
- 如果任务只需要实验参数或运行方式，优先参考现有文档，不要为了“统一风格”去重写历史说明文档。

## Memory Bank Maintenance

- 目录结构、入口脚本、模块归属发生变化时，更新 `memory-bank/architecture.md`。
- 环境、核心依赖、验证命令发生变化时，更新 `memory-bank/tech-stack.md`。
- 常用开发流程、验证路径、建议改动边界发生变化时，更新 `memory-bank/implementation-plan.md`。
- 完成一次实质性工作后，在 `memory-bank/progress.md` 追加记录：
  改了什么、为什么改、如何验证、还有什么风险或待办。

## Local Reference Docs

- `README_experiments.md`: planner-only benchmark 的使用说明。
- `REPORT.md`: 阶段性实验汇报。
- `add_nag2g_after.md`: NAG2G 接入说明和运行示例。
- `project_analysis_report.md`: 某次 benchmark 优化结果分析。
- `1-codingfix.md`: 早期模块规划与历史备注。
