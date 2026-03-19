# Implementation Plan

这个文件不是产品路线图，而是当前仓库推荐的开发路径和验证顺序。

## Standard Change Flow

1. 先判断任务归属哪一层：
   `gp_retro_repr`、`gp_retro_feas`、`gp_retro_obj`、`gp_core`、`gp_retro_nn`、`scripts`、`data`。
2. 先读拥有该行为的文件，再读直接调用它的上层入口脚本。
3. 尽量把改动放在本仓库自己的包装层，不把简单需求直接推到第三方 vendored 目录里。
4. 改动后跑最小可用验证，不要直接上大实验。
5. 如果改动会改变目录结构、使用方法或实验口径，更新 memory bank 和相关说明文档。

## File Ownership Heuristics

- 表示层改动：
  `gp_retro_repr/`
- 可行性、模板修复、动作掩码：
  `gp_retro_feas/`
- 多目标与 route-level scalarization：
  `gp_retro_obj/`
- one-step 候选重排、route score、budget、公平 benchmark：
  `gp_core/`
- NAG2G 子进程协议、推理环境接入：
  `gp_retro_nn/`
- 实验批跑、汇总、画图：
  `scripts/`
- encoder/decoder 预训练部件与未来对齐模块：
  `model/`
  如果任务是基于当前 encoder/decoder 做单步逆合成训练，先读
  `model/RETROSYNTHESIS_PLAN.md`

## Validation Ladder

- 只改表示层：
  `python run_repr_demo.py`
- 改可行性或执行器：
  `python run_feas_demo.py`
- 改多目标或选择器：
  `python run_obj_demo.py`
- 改 StepScorer、GPFitnessEvaluator、leaf 行为、executor 与 one-step 配合：
  `python scripts/scoring_smoke_test.py`
- 改 benchmark 主流程、calls budget、公平对比：
  先跑 `README_experiments.md` 里的 smoke 规模命令，再考虑大批量实验。
- 改 NAG2G 适配：
  先跑 `python scripts/inspect_nag2g_one_step.py ...`
  再跑单目标 `run_real_data_gp.py`
- 改 `model/` 下的单步逆合成训练链路：
  先跑 `python model/train_retrosynthesis.py --stage 1 --batch-size 1 --epochs 1 --limit-train 2 --limit-eval 1 --max-train-steps 1 --device cuda`
  再按需要加上 `--generation-eval-samples 1 --preview-samples 1` 验证生成评估路径。
  训练时默认会把 `train_loss.csv`、`eval_metrics.csv`、`generation_examples.csv`、`loss_curve.svg` 和 `run_config.json` 落到 `model/results/test/`。
  如果要在 Stage 1 结果上继续做 Stage 2，当前脚本已支持 `--init-checkpoint` 做模型权重 warm-start。
- 学习或检查 `model/`：
  优先做静态导入、tokenizer 行为检查、数据字段拆分检查，再做最小训练 smoke。

## Benchmark Invariants

- 不要把 planner-only 预算改成 wall time 或代数，除非用户明确要求改变口径。
- 不要让不同 planner 无意中共享 one-step cache。
- 不要让某个 planner 使用不同的 leaf 逻辑、不同 stock、不同 one-step 候选过滤链路。

## Documentation Rules

- 新增或替换入口脚本后，更新 `memory-bank/architecture.md`。
- 新增环境变量、依赖、运行命令后，更新 `memory-bank/tech-stack.md`。
- 完成一个具体任务后，在 `memory-bank/progress.md` 记录结果和验证方式。
