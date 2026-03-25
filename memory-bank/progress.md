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
- `checkpoints_stage2_full-1` 的 Stage 2 probe 也已跑完：
  `final_model.json` 记录 `global_step = 50000`、`stage = 2`。
- 用与 Stage 1 相同的 `eval.csv + max_eval_batches=128 + generation_eval_samples=32` 配置，对 3 个候选 checkpoint 做了对齐评估：
  1. `Stage 1 final`：`eval_loss = 1.578143`，`generation_exact = 0.0`，平均预测长度约 `106.28`
  2. `Stage 2 step 36000`：`eval_loss = 1.530634`，`generation_exact = 0.0`，平均预测长度约 `99.44`
  3. `Stage 2 final`：`eval_loss = 1.544691`，`generation_exact = 0.0`，平均预测长度约 `103.81`
  当前按验证集 loss 和长度分布看，`Stage 2 step 36000` 是三者里最好的候选，但生成仍明显塌缩，没有出现 exact match。

### 当前判断

- encoder 这边最容易直接污染训练结论的两个问题已经压住了。
- 现在最小 Stage 1 训练闭环、生成评估闭环、结果落盘闭环、正式 Stage 1、warm-start Stage 2、周期保存和独立 checkpoint 测试都已经打通；当前最核心的结论是：
  token-level loss 能有效下降，但生成仍塌缩，下一步重点应转向更有效的阶段切换、评估频率和生成口径，而不是继续只看 CE loss。

## 2026-03-22

### 本次动作

- 为新的 REPA 风格单步逆合成主线新增了数据预处理入口：`model/preprocess_retrosynthesis_data.py`。
- 新脚本读取 `model/data/{train,eval,test}.csv`，对 `product` 和 precursor components 做 RDKit canonicalization，对 precursor components 做确定性排序，并用统一 regex tokenizer 重新生成 token 序列。
- 输出 CSV 同时保留兼容现有训练脚本的 `reactants>reagents>production` 字段，以及后续新训练方案需要的 `product_raw / product_canonical / product_tokens / precursor_raw / precursor_canonical_sorted / precursor_components_canonical / precursor_tokens` 字段。
- 把这个新入口同步记录到了 `memory-bank/architecture.md` 和 `memory-bank/implementation-plan.md`。

### 验证

- 用 `python -m py_compile model/preprocess_retrosynthesis_data.py` 做了静态语法检查。
- 用
  `conda run -n retrogp python model/preprocess_retrosynthesis_data.py --output-dir model/results/preprocess_smoke --limit 8`
  对 train/eval/test 各 8 条样本做了 smoke 验证。
- smoke 结果里三个 split 都是：
  `kept_rows = 8`
  `dropped_rows = 0`
  说明在这批样本上 canonicalization、component 排序和输出字段构造都跑通了。

### 当前判断

- 新的数据预处理脚本已经把“canonicalized precursor-set + 统一 tokenization”这一步单独落地，后续可以在不改原始 `model/data/*.csv` 的前提下，生成新的训练数据版本供 REPA 风格方案使用。
- 这一步暂时只解决了数据表示问题，还没有改新的联合损失训练主线；下一步应继续改 `retro_model.py`、`decoder/model.py`、`train_retrosynthesis.py` 和 `evaluate_checkpoint.py`，把 reactant teacher encoder 和 sequence-level alignment loss 接进去。

### 后续动作

- 已继续把模型和训练侧接到 REPA 风格主线：
  `model/retro_model.py` 新增 `SequenceProjector`、`encode_teacher_reactants()`、mask-aware pooling 和 `forward_repa()`；
  `model/decoder/model.py` 新增可选加权 CE 和 hidden state 返回；
  新增 `model/train_retrosynthesis_repa.py` 和 `model/evaluate_repa_checkpoint.py` 两个独立入口，不再污染旧的 staged baseline 脚本。
- 用
  `python -m py_compile model/retro_model.py model/decoder/model.py model/train_retrosynthesis_repa.py model/evaluate_repa_checkpoint.py`
  做了静态验证。
- 用 processed smoke 数据跑通了新的最小训练闭环：
  `conda run -n retrogp python model/train_retrosynthesis_repa.py --train-csv model/results/preprocess_smoke/train.csv --eval-csv model/results/preprocess_smoke/eval.csv --save-dir model/results/repa_smoke_ckpt --results-dir model/results/repa_smoke --init-checkpoint model/checkpoints_stage2_full-1/model_step_00036000.pt --epochs 1 --batch-size 2 --grad-accumulation 1 --max-train-steps 1 --max-eval-batches 1 --top-decoder-blocks 4 --device cuda`
  输出：
  `trainable_params = 892139520`
  `train_loss = 1.497133`
  `ce_loss = 1.300497`
  `align_loss = 0.983179`
  `eval_loss = 1.649441`
  `eval_ce_loss = 1.512445`
  `eval_align_loss = 0.684981`
- 用
  `conda run -n retrogp python model/evaluate_repa_checkpoint.py --checkpoint model/results/repa_smoke_ckpt/final_model.pt --csv model/results/preprocess_smoke/eval.csv --results-dir model/results/repa_eval_smoke --batch-size 2 --max-eval-batches 1 --generation-eval-samples 2 --preview-samples 1 --device cuda`
  跑通了新的最小评估闭环，确认新 evaluator 能读取 processed CSV，计算 `eval_loss / eval_ce_loss / eval_align_loss`，并写出生成样本。
- 继续把 REPA 主线从“只有 pooled sequence-level 对齐”推进到“sequence-level + token-level 对齐”：
  `model/retro_model.py` 新增 `token_projector` 并把 `forward_repa()` 改成同时输出
  `seq_align_loss / tok_align_loss`；
  `model/train_retrosynthesis_repa.py` 和 `model/evaluate_repa_checkpoint.py` 改成显式接收
  `--seq-align-weight`、`--tok-align-weight`，
  并把 `train_loss.csv / eval_metrics.csv` 的日志字段扩展为
  `train_loss / ce_loss / align_loss / seq_align_loss / tok_align_loss`
  与
  `eval_loss / eval_ce_loss / eval_align_loss / eval_seq_align_loss / eval_tok_align_loss`。
- 在实现 token-level 对齐时发现，teacher encoder 的 tokenizer 会把部分 bracket token 再切成多个子 token，例如 `[O-]` 会被切成 `"[UNK]", "O", "-", "[UNK]"`；
  因此不能简单要求 teacher token 数和 decoder token 数一一相等。
  当前修法是在 collator 里先基于 canonicalized regex token 序列构造 `teacher_group_lengths`，
  再在 `forward_repa()` 里把 teacher hidden 按“每个 SMILES token 对应多少个 encoder 子 token”做分组 mean pooling，
  最后再和 decoder token hidden 对齐。
- 用新的 token-bridge 版本重新跑通了 smoke 训练：
  `conda run -n retrogp python model/train_retrosynthesis_repa.py --train-csv model/results/preprocess_smoke/train.csv --eval-csv model/results/preprocess_smoke/eval.csv --save-dir model/results/repa_tok_smoke_ckpt --results-dir model/results/repa_tok_smoke --init-checkpoint model/checkpoints_stage2_full-1/model_step_00036000.pt --epochs 1 --batch-size 2 --grad-accumulation 1 --seq-align-weight 0.1 --tok-align-weight 0.2 --top-decoder-blocks 4 --max-train-steps 1 --max-eval-batches 1 --device cuda`
  输出：
  `train_loss = 1.578019`
  `ce_loss = 1.281135`
  `align_loss = 0.296884`
  `seq_align_loss = 0.986955`
  `tok_align_loss = 0.990942`
  `eval_loss = 1.823502`
  `eval_ce_loss = 1.563218`
  `eval_align_loss = 0.260284`
  `eval_seq_align_loss = 0.664397`
  `eval_tok_align_loss = 0.969220`
  说明新的 token-level 桥已经完整接通，且不会再因 `[O-]` 这类 bracket token 导致 teacher/decoder token 数不匹配。
- 在正式 `repa_probe` 命令上又遇到了一类更长序列的失败样本：
  早期实现虽然按 `max_reactants_len - 1` 限制了 decoder token 数，但还没有同时保证 teacher encoder 的 subtoken 总数不超过 `max_teacher_len - 2`。
  这会在 `teacher_group_lengths` 累加值超过 teacher 实际保留 subtoken 数时触发：
  `ValueError: teacher subtoken counts do not match grouped lengths`
- 已在 `model/train_retrosynthesis_repa.py` 的 `RepaCollator` 中把 reactant token 选择改成“双预算截断”：
  同时满足
  `decoder token count <= max_reactants_len - 1`
  和
  `teacher subtoken count <= max_teacher_len - 2`。
  这样即使某些 bracket token 会展开成多个 encoder subtokens，也不会再在长样本上超出 teacher 预算。
- 用一个轻量 tokenizer-only 检查脚本对 `model/data_repa_v1/train.csv` 前 `2000` 条样本做了扫描：
  在新的双预算截断规则下，所有样本都满足
  `sum(teacher_group_lengths) == actual_teacher_subtoken_count`。
  这说明这次正式训练里暴露出来的长样本截断 bug 已经被压住。

## 2026-03-25

### 本次动作

- 确认了历史 REPA 权重目录 `model/checkpoints_repa_probe-2/` 是完整可用的一组训练产物，包含：
  `final_model.pt/.json`
  `final_resume.pt/.json`
  `latest_model.pt/.json`
  `latest_resume.pt/.json`
  以及周期保存的 `model_step_00036000.pt/.json`。
- 确认 `model/checkpoints_repa_probe-2/final_model.json` 记录的最终训练步数是
  `global_step = 50652`，因此当前这轮 `repa_probe-2` 的最终模型权重应以
  `final_model.pt` 为主，而不是目录里的中间 step checkpoint。
- 对比了这轮 REPA 已有的 eval 记录：
  `model/results/repa_probe-2/eval_metrics.csv` 中 final 的
  `eval_loss = 1.317595`
  优于
  `model/results/repa_probe-2_checkpoints/step_00036000_eval64/metrics.json` 中 step 36000 的
  `eval_loss = 1.336737`；
  因此这次新的评估与测试继续选用
  `model/checkpoints_repa_probe-2/final_model.pt`。
- 用这份历史权重重新对处理后的 REPA 数据做了新的 dated 结果落盘：
  `model/results/repa_eval_0325_01/`
  `model/results/repa_test_0325_02/`
- 同步把 `model/` 实验输出目录的命名约定写入了 `memory-bank/implementation-plan.md`：
  从现在开始，新实验的 checkpoint/results 目录统一采用
  `MMDD_01`、`MMDD_02`、`MMDD_03`
  这种日期加当日序号的方式区分。

### 验证

- 新的 eval 结果目录
  `model/results/repa_eval_0325_01/metrics.json`
  使用：
  `checkpoint = model/checkpoints_repa_probe-2/final_model.pt`
  `csv = model/data_repa_v1/eval.csv`
  `batch_size = 32`
  `max_eval_batches = 128`
  `generation_eval_samples = 64`
  得到：
  `eval_loss = 1.328984`
  `eval_ce_loss = 1.260443`
  `eval_align_loss = 0.068541`
  `eval_seq_align_loss = 0.050643`
  `eval_tok_align_loss = 0.317385`
  `generation_exact = 0.0`
- 新的 test 结果目录
  `model/results/repa_test_0325_02/metrics.json`
  使用相同 checkpoint 和相同评估口径，在
  `model/data_repa_v1/test.csv`
  上得到：
  `eval_loss = 1.326400`
  `eval_ce_loss = 1.258189`
  `eval_align_loss = 0.068211`
  `eval_seq_align_loss = 0.049906`
  `eval_tok_align_loss = 0.316102`
  `generation_exact = 0.0`
- 两个新结果目录里都已写出：
  `metrics.json`
  `run_config.json`
  `generation_examples.csv`
  可直接回看预测样本。
- 随后又补跑了一组更完整的 full eval / full test 结果，继续使用同一份历史权重
  `model/checkpoints_repa_probe-2/final_model.pt`，
  输出目录为：
  `model/results/repa_eval_0325_03/`
  `model/results/repa_test_0325_04/`
- 这组 full 结果使用：
  `batch_size = 64`
  `max_eval_batches = null`
  `generation_eval_samples = 128`
  也就是 loss 完整跑满整个 split，只把生成 exact-match 保留为 128 条样本抽样。
- full eval 指标为：
  `eval_loss = 1.328951`
  `eval_ce_loss = 1.260346`
  `eval_align_loss = 0.068605`
  `eval_seq_align_loss = 0.051515`
  `eval_tok_align_loss = 0.317269`
  `generation_exact = 0.015625`
  对应 `128` 条生成样本里命中 `2` 条 exact match。
- full test 指标为：
  `eval_loss = 1.331448`
  `eval_ce_loss = 1.262749`
  `eval_align_loss = 0.068699`
  `eval_seq_align_loss = 0.051496`
  `eval_tok_align_loss = 0.317746`
  `generation_exact = 0.0078125`
  对应 `128` 条生成样本里命中 `1` 条 exact match。

### 当前判断

- `repa_probe-2` 这轮历史权重的最佳已知候选仍然是
  `model/checkpoints_repa_probe-2/final_model.pt`。
- 这次新的 quick eval / quick test 在 loss 上比较稳定，eval/test 数值接近，没有出现明显 split gap；
  但 `generation_exact` 依然是 `0.0`，生成样本仍明显存在塌缩或无关长串输出，说明当前 REPA 主线还没有把较低 loss 转化成可用的条件生成质量。
- 补跑 full eval / full test 后，可以更明确地说：
  这份权重在 loss 上的 split gap 很小，
  `full eval_loss = 1.328951`
  `full test_loss = 1.331448`
  差值约 `0.0025`；
  但生成质量仍然偏弱，只是在 `128` 条样本抽样上已经不是完全 `0` 命中，而是出现了极少量 exact match。

### 风险与待办

- 这次 dated 结果目录采用的是快速评估口径：
  `batch_size = 32`
  `max_eval_batches = 128`
  `generation_eval_samples = 64`
  不是完整跑完全部 eval/test split 的 exhaustive 结果。
- 如果后续要写正式实验汇报，建议再对选中的 checkpoint 跑一次 full eval / full test，或者至少增大 `max_eval_batches` 与 `generation_eval_samples` 做更稳的对照。
- 当前 `0325_03 / 0325_04` 已经把 loss 完整跑满了 eval/test split；
  但生成 exact 仍然只是 `128` 条样本抽样，不是对全部 `10w+` 条样本做逐条生成。

## 2026-03-25 Canonical Generation Evaluation Refactor

### 改了什么

- 新增
  `model/smiles_eval.py`
  用 RDKit 对单个分子和 precursor set 做 canonicalization，并在评估时屏蔽无效 SMILES 的 parse log 噪声。
- 调整
  `model/decoder/model.py`
  的 beam search：
  现在支持返回全部 beam 候选；
  同时在 beam 模式下禁用共享 KV cache，避免多分支共用 cache 造成 top-k 候选不可靠。
- 调整
  `model/retro_model.py`
  的 `generate(...)`
  接口，支持 `return_all_beams=True`，把 top-k 候选继续上传给评估逻辑。
- 重写
  `model/train_retrosynthesis.py`
  中的 `evaluate_generation(...)`：
  不再只算 raw string exact；
  现在同时输出
  `generation_exact`
  `generation_topk_exact`
  `generation_raw_exact`
  `generation_invalid_top1_rate`
  并把 canonical target / canonical pred / 全部 beam 候选写进 `generation_examples.csv`。
- 更新
  `model/evaluate_checkpoint.py`
  和
  `model/evaluate_repa_checkpoint.py`
  以保存新的 generation 指标和更完整的样本明细。

### 为什么改

- 之前仓库里的 `generation_exact` 是 raw decoded string 的 top-1 全串相等，这会把“化学等价但写法不同”的预测直接判错。
- 之前 `beam_width > 1` 的实现没有可靠地把全部 beam 候选交给评估端，也不适合作为正式 top-k 指标来源。
- 这次改动把训练期/评估期的表示口径至少统一到了“canonical precursor set”这一层，便于后续再补
  round-trip
  或更正式的 benchmark 汇报。

### 如何验证

- 静态编译通过：
  `python -m py_compile model/smiles_eval.py model/train_retrosynthesis.py model/evaluate_checkpoint.py model/evaluate_repa_checkpoint.py model/retro_model.py model/decoder/model.py`
- 最小运行 smoke：
  在
  `model/checkpoints_repa_probe-2/final_model.pt`
  上，用
  `beam_width = 3`
  对
  `model/data_repa_v1/eval.csv`
  的前 `2` 条样本调用新的 `evaluate_generation(...)`；
  成功返回了新的 generation 指标字典和 canonicalized 预测字段，没有再打印 RDKit parse error。

### 风险与待办

- 目前只完成了 canonical top-k exact 这一层，还没有把
  round-trip accuracy
  接进来。
- `10w+` 样本规模下的 beam search 评估成本很高；这次没有额外落新的正式 `0325_05/06` 结果目录，后续如果要跑标准 top-k，请先控制
  `max_eval_batches`
  和
  `generation_eval_samples`
  再决定是否做更大的正式评估。

## 2026-03-25 CUDA Eval OOM Mitigation

### 改了什么

- 在
  `model/train_retrosynthesis.py`
  中新增了评估期 autocast helper：
  CUDA 下默认自动选
  `bf16`
  或
  `fp16`
  做 loss 评估。
- `evaluate_loss(...)`
  和
  `evaluate_repa_loss(...)`
  现在都支持 `amp_dtype`，默认在 CUDA 上启用低精度评估，减少显存占用。
- `model/evaluate_checkpoint.py`
  和
  `model/evaluate_repa_checkpoint.py`
  新增
  `--amp-dtype {fp32,fp16,bf16}`
  参数；不传时也会自动走 CUDA 低精度评估。

### 为什么改

- 用户按
  `batch_size = 64`
  跑
  `model/evaluate_repa_checkpoint.py`
  时，在
  `model/encoder/encoders.py`
  的 encoder 前向阶段直接触发了 CUDA OOM。
- 现场检查时，4090 上只有约
  `18GB`
  空闲显存，而 full-precision 的 REPA loss 评估配
  `batch_size = 64`
  显存压力过高。

### 如何验证

- 静态编译通过：
  `python -m py_compile model/train_retrosynthesis.py model/train_retrosynthesis_repa.py model/evaluate_checkpoint.py model/evaluate_repa_checkpoint.py`
- smoke 验证：
  用原始用户命令的核心配置
  `device=cuda`
  `batch_size=64`
  `max_eval_batches=1`
  `generation_eval_samples=0`
  再加
  `--amp-dtype bf16`
  重跑后成功完成，输出：
  `eval_loss = 1.353245`
  `eval_ce_loss = 1.281749`
  `eval_align_loss = 0.071496`
  没有再触发 CUDA OOM。

### 风险与待办

- 低精度评估会让 loss 数值有轻微浮动，不应和历史 full-precision 小数点后很多位做严格逐位比较。
- 即使启用 autocast，`generation_beam_width = 10`
  的大规模正式评估仍然耗时很高；显存问题缓解了，不代表 top-k 评估成本低。

## 2026-03-25 Smoke Artifact Cleanup

### 本次动作

- 按用户要求，删除了 `model/` 下所有名称带 `smoke` 的历史开发测试产物，覆盖：
  `model/checkpoints_smoke`
  `model/checkpoints_smoke_preview`
  `model/checkpoints_results_smoke`
  `model/results/preprocess_smoke`
  `model/results/periodic_smoke`
  `model/results/checkpoint_smoke`
  `model/results/repa_smoke`
  `model/results/repa_smoke_ckpt`
  `model/results/repa_eval_smoke`
  `model/results/repa_tok_smoke`
  `model/results/repa_tok_smoke_ckpt`
  `model/results/repa_tok_eval_smoke`

### 验证

- 重新扫描
  `find model -maxdepth 3 | rg 'smoke'`
  后已无任何匹配项，说明 `model/` 下 smoke 目录已清空。

### 说明

- 这次删除只清理开发 smoke 产物，没有动
  `repa_probe`
  `stage1_full`
  `stage2_full`
  等正式或半正式实验目录。

## 2026-03-26 REPA `CC` Collapse Analysis

### 本次动作

- 针对
  `model/checkpoints_repa_probe-2/final_model.pt`
  在
  [repa_eval_0325_05](/data1/ytg/retrogp/model/results/repa_eval_0325_05)
  中出现的大量
  `CC`
  预测，拆成四块做排查：
  训练分布、token 频率、解码偏置、teacher 对齐是否有效传到 decoder。
- 统计了
  `model/data_repa_v1/{train,eval,test}.csv`
  的目标分布和前缀分布，重点看：
  exact target `CC`
  比例、
  首 token `C`
  比例、
  前两 token 为
  `C C`
  的比例。
- 统计了
  [repa_eval_0325_03/generation_examples.csv](/data1/ytg/retrogp/model/results/repa_eval_0325_03/generation_examples.csv)
  和
  [repa_eval_0325_05/generation_examples.csv](/data1/ytg/retrogp/model/results/repa_eval_0325_05/generation_examples.csv)
  的预测长度、前缀分布和 beam 候选分布。
- 对单样本做了 logits 探针，对比
  “有 product 条件”
  和
  “无 encoder 条件”
  的前两步 top logits，确认条件信息确实在影响 decoder，但强度不足以压过
  `C -> C`
  的语言模型先验。

### 关键发现

- 数据分布确实强烈偏向碳开头，但不能单独解释当前塌缩：
  train/eval/test 上 exact target
  `CC`
  只占约
  `4.4%`，
  但前两 token 为
  `C C`
  的样本约占
  `44.8% ~ 45.0%`。
  这说明
  `CC`
  作为“前缀”非常常见，但作为“完整答案”并不常见。
- 生成结果明显过度放大了这个前缀先验：
  在
  [repa_eval_0325_05/generation_examples.csv](/data1/ytg/retrogp/model/results/repa_eval_0325_05/generation_examples.csv)
  里，
  `128/128`
  条 top-1 预测都以
  `C`
  开头，
  `121/128`
  以
  `CC`
  开头，
  `116/128`
  直接等于
  `CC`；
  但同一批
  `128`
  条目标里只有
  `6`
  条真值是
  `CC`。
- beam search 的当前重排策略会进一步放大短序列偏置：
  [model.py](/data1/ytg/retrogp/model/decoder/model.py#L382)
  到
  [model.py](/data1/ytg/retrogp/model/decoder/model.py#L391)
  在最终 rerank 时对每个生成 token 额外减去
  `0.2`，
  这会系统性偏好更短输出。
  直接体现在：
  `0325_03`
  的平均预测长度约
  `34.0`
  字符，
  `0325_05`
  降到约
  `3.48`
  字符，
  同时 exact
  `CC`
  预测从
  `33/128`
  激增到
  `116/128`。
- 条件信息不是完全没进 decoder，但目前只是在“修边”，没有改变主导 token 先验：
  单样本 logits 探针显示，无 encoder 条件时 decoder 在前两步会给
  `C`
  极高分，同时还混入
  `(`、
  `/`、
  `[Si]`
  等明显属于预训练语言模型先验的 token；
  加上 product 条件后，这些噪声 token 会被压下去，`O`
  和
  `[EOS]`
  的分数会抬高，但 top-1 依然常常保持
  `C`
  或
  `C -> C`。
- teacher 对齐项已经在训练里生效，但它没有直接约束 next-token logits：
  [retro_model.py](/data1/ytg/retrogp/model/retro_model.py#L193)
  到
  [retro_model.py](/data1/ytg/retrogp/model/retro_model.py#L214)
  的 REPA 辅助项是对最后几层 decoder hidden 做
  sequence/token cosine 对齐，
  再经过 projector 参与总 loss；
  它优化的是表示空间，不是直接对生成分布做校正。
  当前 quick eval 里
  `eval_ce_loss ≈ 1.2629`
  而加权后的
  `eval_align_loss ≈ 0.0684`，
  辅助项量级远小于 CE，说明它更像弱正则，而不是能主导解码行为的目标。

### 当前判断

- `CC`
  塌缩不是单一原因，而是三件事叠加：
  1. 数据前缀本身强烈偏向
  `C -> C`
  2. decoder 预训练先验很强，条件信息只造成有限重排
  3. beam search 的长度惩罚把这种前缀偏置进一步放大成“尽快停在短串”
- 所以目前不能把问题简单归咎为
  “teacher 对齐完全无效”；
  更准确的说法是：
  teacher 对齐让 decoder 更像 reactant 表示了，但还没有强到足以显著提升 product-conditioned next-token control。

### 待办

- 下一步优先修 beam search 的长度重排，去掉或重设
  `-0.2 * length`
  这类强短序列惩罚，再重新测
  top-k。
- 在修完解码偏置后，再判断是否需要继续动训练：
  例如减弱 decoder 语言模型先验、强化 cross-attn 可训练范围、或提高与条件控制直接相关的训练信号。
