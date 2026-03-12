# Progress

## 2026-03-12

### 本次动作

- 初始化了项目级 `AGENTS.md`。
- 新建了 `memory-bank/`，把项目背景、技术栈、架构地图、开发流程和进度记录固定下来。
- 调整了 `.gitignore`，让 `AGENTS.md` 和 `memory-bank/*.md` 可以被 Git 跟踪。
- 对当前仓库做了源码级梳理，覆盖了核心包、入口脚本、数据目录、实验脚本和内嵌第三方目录。
- 新识别并记录了 `model/` 目录：其中包含一个 BERT 风格 encoder、一个 GPT 风格 decoder，以及配套反应数据集。
- 确认 `retrogp` 环境里已经有 `torch`、`transformers`、`tokenizers`、`PyYAML`，并把缺失的直接依赖同步回 `environment.yml`。
- 把 `environment.yml` 从旧的 UTF-16LE 文本转成了 UTF-8，后续可以正常维护。

### 当前判断

- 当前主线不是早期 demo，而是：
  `run_real_data_gp.py` + `scripts/benchmark_planners.py`
- 当前最关键的核心源码是：
  `gp_core/scoring.py`
  `gp_core/search.py`
  `gp_core/budget.py`
  `gp_retro_feas/executor.py`
  `gp_retro_repr/stock.py`
  `gp_retro_nn/nag2g_subprocess.py`
- 仓库的上下文比较分散，现有说明主要散落在：
  `README_experiments.md`
  `REPORT.md`
  `add_nag2g_after.md`
  各模块 README

### 风险与提醒

- 仓库里已经 vendored 了多个上游项目，默认不要把本项目需求直接实现到 `ASKCOS/`、`ChemProjector-main/`、`NAG2G-main/`、`scscore/` 里。
- 当前仓库已有本地未提交改动：
  `gp_core/config.py`
  `gp_core/data_loading.py`
  后续如果任务碰到这两个文件，先重新检查 `git status`，不要覆盖已有工作。
- `model/decoder/model.py` 目前导入会卡在 `from utils.train_utils import Variable`；这是项目内缺少模块，不是 `pip` 包缺失。
- `model/encoder` 和 `model/decoder` 词表 token 集基本兼容，但大多数普通 token 的 id 不同，后续不能直接假设“同 token 同 id”。

### 尚未做的事

- 还没有把分散的历史说明整合成一个统一顶层 `README.md`。
- 还没有补专门的自动化测试目录；目前主要依赖 demo 和 `scripts/scoring_smoke_test.py`。
- 这次只做了记忆与文档配置，没有修改任何业务逻辑代码，也没有运行耗时实验。
