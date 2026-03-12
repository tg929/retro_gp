# Tech Stack

## 语言与环境

- 主仓库环境由 `environment.yml` 描述，环境名是 `retrogp`。
- `environment.yml` 当前是 UTF-16LE 编码；命令行查看时需要 `iconv -f UTF-16LE -t UTF-8 environment.yml`。
- NAG2G 独立环境由 `nag2g.yml` 描述，环境名是 `nag2g`。
- 常见运行方式是：
  主脚本在 `retrogp` 环境执行，NAG2G one-step 推理由 `--nag2g-python` 指向单独环境中的 Python。

## 核心依赖

- 化学表示与模板应用：`rdkit`、`rdchiral`
- 数值与深度学习：`numpy`、`torch`
- 配置和数据处理：`PyYAML`、`csv`、`pandas`
- 图与分析辅助：`networkx`、`matplotlib`
- 合成复杂度：vendored `scscore/` 与本地 `multi-score/`

## 代码分层

- `gp_retro_repr/`: 问题表示层
- `gp_retro_feas/`: 可行性与约束层
- `gp_retro_obj/`: 多目标与选择层
- `gp_core/`: GP 搜索、step/route scoring、budget、baseline planners
- `gp_retro_nn/`: one-step 模型接口和 NAG2G 子进程适配

## 外部内嵌项目

- `NAG2G-main/`: one-step retrosynthesis 模型本体
- `ASKCOS/`: leaf/stop 逻辑和历史参考来源之一
- `ChemProjector-main/`: building block 处理与相关实验资产
- `scscore/`: SCScore upstream 代码和模型

## 主要命令

- 纯本地 smoke test:
  `python scripts/scoring_smoke_test.py`
- 结构/执行 demo:
  `python run_repr_demo.py`
  `python run_feas_demo.py`
  `python run_obj_demo.py`
- planner-only benchmark:
  参考 `README_experiments.md`
- NAG2G one-step 结果检查:
  `python scripts/inspect_nag2g_one_step.py ...`

## 输出目录

- `logs/benchmarks/`: benchmark CSV 与中间结果
- `logs/run_real_data_gp_output_*.txt`: 历史运行日志
- `setting/save_vars.py` 生成的配置快照默认也写到 `logs/`
