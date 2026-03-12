# Project Context

## 项目目标

这个仓库是一个面向逆合成规划的研究型代码库，核心目标有两条：

- 用 `gp_retro_repr -> gp_retro_feas -> gp_retro_obj -> gp_core` 这条分层链路，把多步 retrosynthesis 表示、执行、打分和搜索拆成可维护模块。
- 在固定 one-step 模型为 NAG2G、固定 stock/leaf 判定的条件下，做 `GP / greedy / beam / best_first` 的 planner-only 公平对比。

## 当前主线

- `run_real_data_gp.py` 是当前项目内最接近“主入口”的脚本。
  它从 `data/target molecular/config.yaml` 读取命名目标，加载 building blocks 和 templates，按需挂接 NAG2G，并运行 GP 搜索。
- `scripts/benchmark_planners.py` 是当前最成熟的实验框架。
  它把 GP 和多个 baseline planner 放在同一个 calls budget 下比较，输出 `per_target.csv` 和 `summary.csv`。
- `gp_retro_nn/` 提供 NAG2G 的持久子进程接入，让主环境和 one-step 推理环境解耦。

## 关键设计约束

- 预算统一使用 one-step 模型的真实推理调用次数，而不是 wall time 或进化代数。
- 候选扩展要尽量共用一套 one-step 过滤和重排逻辑，核心实现是 `gp_core/scoring.py` 的 `StepScorer`。
- 路线的 solved 判定依赖 `Inventory.is_leaf()`，默认等于 purchasable，也可扩展为 ASKCOS 风格的 small/popular leaf。
- 模板加载默认从 `data/reaction_template/hb.txt` 读取，并按 retro 方向整理。

## 仓库现状

- 已有模块化源码、demo、benchmark、实验说明和多份历史分析文档。
- 没有统一的顶层 README；上下文分散在 `README_experiments.md`、各模块 README 和若干阶段性笔记中。
- 仓库内包含多个内嵌第三方项目：`ASKCOS/`、`ChemProjector-main/`、`NAG2G-main/`、`scscore/`。

## 修改优先级

- 优先在本仓库自己的模块中实现行为变化。
- 只有在接口不够用时，才考虑深入 vendored third-party 目录。
- 修改前先确认任务属于哪一层，避免把“表示层问题”改进“搜索层”或者把“实验参数问题”改成“模型代码问题”。
