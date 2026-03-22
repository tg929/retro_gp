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
- 确认 `model/decoder/model.py` 中的 `utils.train_utils.Variable` 只在这一处出现，属于孤立的历史遗留导入。
- 已删除 `model/decoder/model.py` 中未使用的 `utils.train_utils.Variable` 导入，decoder 代码现在可以直接被导入。
- 确认 `model/encoder/encoders.py` 当前默认 tokenizer 设置会 lower-case SMILES；显式传 `do_lower_case=False` 后才符合化学语义。
- 确认重新上传后的 `model/encoder/.../checkpoint.pt` 和 `model/decoder/.../SMILES-650M-3B-Epoch1.pt` 都可以被 `torch.load` 正常读取，且 decoder 权重与当前 GPT 结构严格匹配。
- 新增 `model/RETROSYNTHESIS_PLAN.md`，把当前 encoder、decoder 和 `model/data/` 的单步逆合成实施路径整理成一份可执行计划书。
- 同步修正了 `memory-bank/architecture.md` 中关于 `model/decoder` 导入问题和权重状态的过期描述。

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
- `model/decoder/model.py` 的孤立历史导入已经移除；如果后续还有导入错误，需要按新的实际 traceback 重新判断。
- `model/encoder` 和 `model/decoder` 词表 token 集基本兼容，但大多数普通 token 的 id 不同，后续不能直接假设“同 token 同 id”。
- 当前已经可以做“真实预训练权重前向”级别的静态学习与 shape 验证；但还没有开始任何训练或生成实验。

### 尚未做的事

- 还没有把分散的历史说明整合成一个统一顶层 `README.md`。
- 还没有补专门的自动化测试目录；目前主要依赖 demo 和 `scripts/scoring_smoke_test.py`。
- 这次只做了记忆与文档配置，没有修改任何业务逻辑代码，也没有运行耗时实验。

## 2026-03-19

### 本次动作

- 完成了单步逆合成训练 `Stage 0` 的两个硬修复：
  `model/encoder/encoders.py` 现在显式使用 `do_lower_case=False`
  `model/encoder/local_bert.py` 现在会把传入的 `attention_mask` 真正送进 self-attention
- 保持了 encoder 代码改动最小，只修训练前必须修的两处行为问题，没有提前引入训练脚本或额外抽象层。
- 同步更新了 `memory-bank/architecture.md`，把 `model/encoder` 相关状态改成当前真实实现。
- 在 `model/decoder/model.py` 中加入了可选 cross-attention，并把条件 memory 透传到了 `forward()`、`generate()`、`beam_search_generate()` 路径。
- 新增 `model/retro_model.py`，把 frozen encoder、轻量 aligner 和 conditional decoder 组装成一个最小可训练模型。
- 新增 `model/train_retrosynthesis.py`，实现了 CSV 数据集读取、collator、Stage 1 / Stage 2 冻结策略、训练循环、loss eval、生成预览和 checkpoint 保存。

### 验证

- 用真实 encoder checkpoint 和 `model/data/train.csv` 第一条 product 验证 tokenizer 输出，确认 `C/O/N` 等大写 token 被保留，没有再被 lower-case。
- 用同一条 product 做“单独前向”和“带 padding 前向”的对比，非 pad 位置输出最大差为 `3.052e-05`，说明 padding mask 已经生效。
- 在真实权重上完成了 2 条样本的组合模型前向和反向，得到：
  `logits_shape = (2, 60, 591)`
  `loss = 5.316456`
  `self_attn_layers = 13`
  并确认 Stage 1 可训练参数已经拿到梯度。
- 跑通了 `python model/train_retrosynthesis.py --stage 1 --batch-size 1 --epochs 1 --limit-train 2 --limit-eval 1 --max-train-steps 1 --device cuda ...`
  smoke test，输出：
  `trainable_params = 227874816`
  `train_loss = 5.283941`
  `eval_loss = 6.767875`
  并成功保存 `model/checkpoints_smoke/best.pt`
- 跑通了带生成评估的 smoke test：
  `python model/train_retrosynthesis.py --stage 1 --batch-size 1 --epochs 1 --limit-train 2 --limit-eval 1 --max-train-steps 1 --generation-eval-samples 1 --preview-samples 1 --device cuda ...`
  输出：
  `train_loss = 4.627955`
  `eval_loss = 5.943709`
  `generation_exact = 0.000000`
  并打印了一条 `product / target / pred` 预览样本，确认生成评估路径已接通；对应 checkpoint 已保存到 `model/checkpoints_smoke_preview/best.pt`
- 跑了一个更像样的 Stage 1 子集实验：
  `python model/train_retrosynthesis.py --stage 1 --batch-size 2 --epochs 1 --limit-train 32 --limit-eval 4 --max-train-steps 10 --generation-eval-samples 2 --preview-samples 2 --device cuda ...`
  输出显示：
  `train_loss` 从 `5.608910` 降到 `3.008404`
  `eval_loss = 2.573587`
  `generation_exact = 0.000000`
  预览生成仍然塌缩为长串重复的 `c`，说明当前 Stage 1 在极小步数下已经学到 token-level loss，但还没有学到可用的条件化生成行为。
- 又跑了一个更长的 Stage 1 子集实验：
  `python model/train_retrosynthesis.py --stage 1 --batch-size 2 --epochs 1 --limit-train 256 --limit-eval 16 --max-train-steps 50 --generation-eval-samples 4 --preview-samples 2 --device cuda ...`
  输出显示：
  `train_loss` 从 `5.539189` 降到 `2.372203`
  `eval_loss = 2.487176`
  `generation_exact = 0.000000`
  预览生成依然是长串重复的 `c`，说明仅靠当前 Stage 1 设置，在 50 步量级内还没有把“条件通道可训练”转化成“可用的条件化生成”。
- 跑了一个 Stage 2 小规模对照实验：
  `python model/train_retrosynthesis.py --stage 2 --trainable-decoder-blocks 4 --batch-size 2 --epochs 1 --limit-train 128 --limit-eval 8 --max-train-steps 20 --lr 1e-4 --decoder-lr 1e-5 --generation-eval-samples 2 --preview-samples 2 --device cuda ...`
  输出显示：
  `trainable_params = 429307904`
  `eval_loss = 2.729984`
  `generation_exact = 0.000000`
  预览生成仍然塌缩为长串重复的 `c`，说明在当前小样本、短步数设置下，提前解冻顶部 4 层 decoder 也没有立刻带来可见的条件化生成改善。
- 继续增强了 `model/train_retrosynthesis.py` 的结果落盘能力，新增 `--results-dir`，训练时会把 step 级 `train_loss.csv`、epoch 级 `eval_metrics.csv`、生成预览 `generation_examples.csv`、`loss_curve.svg` 和 `run_config.json` 一起写出。
- 用 `python model/train_retrosynthesis.py --stage 1 --batch-size 1 --epochs 1 --limit-train 2 --limit-eval 1 --max-train-steps 1 --generation-eval-samples 1 --preview-samples 1 --device cuda --results-dir /data1/ytg/retrogp/model/results/test ...`
  验证了结果目录落盘逻辑，确认上述五个文件都会生成，且 `generation_examples.csv` 已能直接看到 `product / decoder_input / target / pred / match`。
- 跑完了一轮正式的 Stage 1：
  `python model/train_retrosynthesis.py --stage 1 --batch-size 2 --epochs 1 --max-train-steps 5000 --lr 1e-4 --decoder-lr 1e-5 --max-eval-batches 128 --generation-eval-samples 8 --preview-samples 4 --device cuda --save-dir /data1/ytg/retrogp/model/checkpoints_stage1_formal --results-dir /data1/ytg/retrogp/model/results/test`
  结果为：
  `eval_loss = 1.660985`
  `generation_exact = 0.0`
  train loss 从前 5 步均值 `4.665` 降到最后 50 步均值 `1.760`
  但生成仍明显塌缩，尚未得到可用的条件化 reactants 输出。
- 给 `model/train_retrosynthesis.py` 增加了最小 warm-start 能力，新增 `--init-checkpoint`，可从已有模型权重继续起下一阶段训练。
- 从 `checkpoints_stage1_formal/best.pt` 提炼出只含模型参数的 `checkpoints_stage1_formal/model_only.pt`，避免后续 warm-start 每次都加载 9.6G 的完整优化器状态。
- 按推荐矩阵先跑了 `test1`：
  `python model/train_retrosynthesis.py --init-checkpoint /data1/ytg/retrogp/model/checkpoints_stage1_formal/model_only.pt --stage 2 --trainable-decoder-blocks 4 --batch-size 2 --epochs 1 --max-train-steps 2000 --lr 5e-5 --decoder-lr 5e-6 --max-eval-batches 128 --generation-eval-samples 8 --preview-samples 4 --device cuda --save-dir /data1/ytg/retrogp/model/checkpoints_stage2_test1 --results-dir /data1/ytg/retrogp/model/results/test1`
  结果为：
  `eval_loss = 1.683021`
  `generation_exact = 0.0`
  生成分布相比纯 Stage 1 末尾稍有变化，不再只是单一长串 `c`，但仍然严重塌缩，离可用输出还有明显距离。
- 继续增强了训练/测试解耦能力：
  `model/train_retrosynthesis.py` 新增 `--save-every-steps`
  会周期性覆盖保存 `latest_model.pt` 和 `latest_model.json`
  供训练过程中并行读取最新权重。
- 新增 `model/evaluate_checkpoint.py`，可独立加载任意保存下来的 checkpoint，对指定 CSV 跑 eval loss 和生成预览，不依赖训练进程继续运行。
- 用一组最小 smoke 验证了这条链路：
  先用 `--save-every-steps 1` 跑了 1 step 训练，成功生成 `checkpoints_periodic_smoke/latest_model.pt`
  再用 `python model/evaluate_checkpoint.py --checkpoint ... --csv model/data/eval.csv --limit 1 ...`
  成功输出 `eval_loss = 6.638617`
  `generation_exact = 0.0`
  并把测试结果写到 `model/results/checkpoint_smoke/`
- 按最新确定的 full-data 工作流，进一步修改了 `model/train_retrosynthesis.py`：
  新增 `--disable-eval`，支持训练主进程只做训练、不做训练内验证；
  `--save-every-steps` 现在会按 `model_step_XXXXXXXX.pt/json` 追加保存 model-only checkpoint，同时继续维护 `latest_model.pt/json`；
  训练结束固定写出 `final_model.pt/json`。
- 同步调整了 `model/evaluate_checkpoint.py` 默认评估集合，从 `model/data/test.csv` 改为 `model/data/eval.csv`，更符合“训练中途评估不用测试集”的当前策略。
- 用 `python -m py_compile model/train_retrosynthesis.py model/evaluate_checkpoint.py`
  验证了这轮改动的语法正确性；本轮没有启动新的训练。
- 在训练被系统 `Killed` 后，又补上了双轨 checkpoint 和断点续训支持：
  `model/train_retrosynthesis.py` 新增 `--resume-from` 和 `--resume-every-steps`；
  高频 `model_step_XXXXXXXX.pt` 继续作为评估用 model-only checkpoint，
  低频 `resume_step_XXXXXXXX.pt` / `latest_resume.pt` / `final_resume.pt` 则保存 model + optimizer + 续训元信息；
  训练集 DataLoader 改为按 `seed + epoch` 的确定性打乱，resume 时会按保存下来的 `epoch_step` 跳过当前 epoch 已处理的 batch，尽量恢复到被 kill 前的位置。
- 用 `python -m py_compile model/train_retrosynthesis.py model/evaluate_checkpoint.py`
  再次验证了 resume 相关改动的语法正确性；本轮仍未启动新的训练。
- 针对被系统 `Killed` 的旧 full-data Stage 1 运行，又整理出了一套可汇报材料，输出到 `model/results/test-full_report/`：
  包含 `train_loss_curve.svg`、`checkpoint_train_loss.svg`、`checkpoint_summary.csv`、`summary.json` 和 `report.md`。
  这套材料基于现有的 `model/results/test-full/train_loss.csv`、`model/checkpoints_stage1_full/model_step_*.json` 和 `model/results/test-full_checkpoints/latest_fast/` 汇总生成；
  关键结论是：训练 loss 在 129157 个已记录 step 内持续下降，但最后成功保存的 checkpoint 仍停在 108000 step；最新 fast eval 的 `eval_loss = 1.6517`、`generation_exact = 0.0`，生成结果仍明显塌缩。
- 为了先做最小推理修正，调整了 `model/decoder/model.py` 的停止条件：
  普通 retrosynthesis 生成和 beam search 不再把 `SEP` 作为停止 token，只在生成到 `EOS` 时停止；
  `linker` 分支保持原有 `SEP` 终止逻辑不变。
  这样训练目标里的 `reactants + [EOS]` 与推理阶段的停止条件更一致，先尽量减轻“经常打满长度上限”的问题。
- 用 `python -m py_compile model/decoder/model.py model/retro_model.py model/train_retrosynthesis.py model/evaluate_checkpoint.py`
  验证了这次停止条件调整没有引入语法错误；本轮没有启动新训练或新评估。
- 调整了仓库根目录 `.gitignore` 对 `model/` 的规则：
  现在 `model/` 下默认放开提交，只继续忽略 `model/data/**` 和各类权重文件（如 `model/**/*.pt` / `.pth` / `.ckpt` / `.safetensors`）；
  后续又进一步收紧为：
  `model/results/**` 不进入 Git，checkpoint 目录下的 `.json` 元信息不进入 Git；
  同时补充忽略 `model/**/.DS_Store`、`model/**/__pycache__/` 和 `model/**/*.pyc`，避免运行噪音文件进入 Git。
- `checkpoints_stage1_full-1` 这一轮 full-data Stage 1 已完整跑完 1 个 epoch：
  `final_model.json` / `final_resume.json` 里记录的 `global_step = 405238`。
- 对这轮完整 Stage 1 的最终权重做了独立评估，结果写到 `model/results/test-full-1_checkpoints/final_eval32/`：
  使用 `eval.csv`、`max_eval_batches = 128`、`generation_eval_samples = 32`。
  结果为：
  `eval_loss = 1.578143`
  `generation_exact = 0.0`
  生成长度仍明显塌缩，32 条样本的平均预测长度约 `106.28`，其中大量样本仍以长串 `C` 开头并接近长度上限。
- 在做这次评估前，再次发现并清理了会占 GPU 显存的异常 `/tmp/.cache/python` 进程及恢复的 `/var/tmp/.crond/upd` crontab 持久化；
  清理后 GPU 上只剩正常的小型系统进程，再进行评估。

### 当前判断

- encoder 这边最容易直接污染训练结论的两个问题已经压住了。
- 现在最小 Stage 1 训练闭环、生成评估闭环、结果落盘闭环、正式 Stage 1、warm-start Stage 2、周期保存和独立 checkpoint 测试都已经打通；当前最核心的结论是：
  token-level loss 能有效下降，但生成仍塌缩，下一步重点应转向更有效的阶段切换、评估频率和生成口径，而不是继续只看 CE loss。
