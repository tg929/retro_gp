# Retrosynthesis Plan

本计划书只针对 `/data1/ytg/retrogp/model/` 目录下的 encoder、decoder 和 `/data1/ytg/retrogp/model/data/` 数据集，目标是在当前仓库内做出一个可训练、可评估、可解释的单步逆合成条件生成模型。

## 1. Goal

目标任务：

- 输入：单一 `product`
- 输出：对应上一步 `reactants`
- 训练数据：`model/data/train.csv`、`model/data/eval.csv`、`model/data/test.csv`
- 训练目标：token-level next-token cross entropy

明确范围：

- encoder 使用当前 `model/encoder/` 下的 BERT 风格模型
- decoder 使用当前 `model/decoder/` 下的 GPT 风格模型
- 不引入第三方新模型作为主干
- 不在本阶段解决多步路线规划，只解决 one-step retrosynthesis

## 2. Current Facts

当前已经确认的事实：

- encoder 结构为 `24 layers / 32 heads / 2048 hidden`
- decoder 结构为 `13 layers / 32 heads / 2048 hidden`
- encoder 与 decoder 的 hidden dim 都是 `2048`
- decoder 当前没有 cross-attention，只是纯自回归 GPT
- encoder 与 decoder 的词表 token 集相近，但大多数普通 token 的 id 不同，不能做 id 级直接对齐
- 数据集右侧 `product` 当前全是单一产物，没有 multi-product 样本
- 数据集表头保留了 `reactants>reagents>production` 命名，但当前样本都表现为 `reactants >> product`，中间 `reagents` 为空
- 左侧 `reactants` 常常包含多个组分，使用 `.` 分隔
- encoder 权重和 decoder 权重都已经能被 `torch.load` 正常加载

当前必须先承认的工程问题：

- `model/encoder/encoders.py` 当前构造 tokenizer 时没有显式传 `do_lower_case=False`，会破坏化学 token 语义
- `model/encoder/local_bert.py` 当前 attention 实现没有实际使用传入的 `attention_mask`
- encoder tokenizer 目前更依赖已经空格切好的输入；decoder tokenizer 才是完整 regex tokenizer
- decoder 当前代码路径里没有“接收 encoder 条件表示”的入口

## 3. Recommended Target Architecture

推荐目标结构：

1. `product` 进入 frozen encoder，输出 `H_enc`
2. `H_enc` 经过轻量 aligner，得到 `M`
3. decoder 输入是 `reactants_prefix`
4. decoder 先做 causal self-attention
5. decoder 每层再对 `M` 做 cross-attention
6. 输出 logits，对真实 `reactants_next_token` 做 cross entropy

形式化表示：

```text
product -> encoder -> H_enc (B, T_p, 2048)
H_enc -> aligner -> M (B, T_p, 2048)

reactants_prefix -> decoder self-attn
decoder hidden + M -> cross-attn
conditioned hidden -> logits -> CE
```

为什么选这个结构：

- 维度天然对齐，hidden size 都是 `2048`
- one-step retrosynthesis 需要 token-level 条件信息，不适合只传一个 pooled vector
- cross-attention 比单纯 prefix/prompt 更适合长 product 和局部结构依赖
- 这是在当前预训练部件基础上成功率最高、解释性最强的方案

明确不推荐的主方案：

- 不推荐只拿 encoder 最前面的 embedding 输出做条件
- 不推荐只把 `[CLS]` 或 `pooled_output` 当唯一条件
- 不推荐一上来全量微调 GPT

## 4. Training Strategy

### Stage 0: Preconditions

先处理会直接干扰训练结论的前置问题：

- 修正 encoder tokenizer 的 lowercasing 问题
- 修正 encoder padding mask 未生效的问题
- 统一训练数据输入规范：
  product 侧明确采用“空格切好 token”还是“重写 encoder tokenizer 支持 raw SMILES”
- 明确 reactants 目标字符串的标准化方案：
  是否做 canonicalize
  是否对 `.` 分隔的组分做稳定排序

Stage 0 的验收条件：

- encoder 能在 batch + padding 条件下稳定前向
- encoder 与 decoder 的 tokenizer 行为有明确文档
- 一条样本经过数据管线后，product/reactants 的输入输出张量定义明确

### Stage 1: Condition Path Warm-Up

只训练条件通道，不急着大改 decoder 主体。

训练参数：

- encoder：冻结
- decoder backbone：冻结
- aligner：训练
- 新增 cross-attention：训练
- 可选放开 `ln_f` 和输出 `head`

这阶段的目标：

- 先把“encoder -> decoder”的条件信息通路训通
- 让 decoder 能读懂 product memory
- 避免 GPT 一开始被随机初始化的条件接口扰乱

训练方式：

- teacher forcing
- decoder input：`[BOS] + reactants[:-1]`
- decoder target：`reactants + [EOS]`
- loss：token-level CE，忽略 PAD

Stage 1 的验收条件：

- 能在一个小 batch 上明显过拟合
- 小规模子集上 `train loss` 稳定下降
- 生成结果不是完全无条件乱写
- attention 行为可解释：decoder 至少开始使用 product memory

### Stage 2: Conditional Decoder Adaptation

在 Stage 1 通路稳定后，再逐步解冻 decoder 上层。

训练参数建议：

- encoder：继续冻结
- aligner：继续训练
- cross-attention：继续训练
- decoder 上层 4 到 6 层：解冻
- `ln_f` 和输出 `head`：训练

这阶段的目标：

- 让 GPT 真正学会利用 product 条件来生成 reactants
- 保留底层 SMILES 语言能力，同时适应 retrosynthesis 任务

训练策略建议：

- 预训练参数用更小学习率
- 新增模块用更大学习率
- 先解冻上层，验证稳定后再决定是否全量解冻

Stage 2 的验收条件：

- `eval loss` 明显优于 Stage 1
- beam search 结果的 top-k exact match 有实际提升
- 输出对 product 有条件依赖，而不是模板化重复生成

### Stage 3: Evaluation And Stabilization

完成正式评估与必要的轻量消融。

必须产出的评估：

- token CE
- top-1 exact match
- top-5 exact match
- 合法 SMILES 比例
- 失败样本类型归因

建议额外消融：

- 不带 cross-attention 的 decoder baseline
- 只用 pooled vector 的简化条件 baseline
- Stage 1 停止 vs Stage 2 解冻上层的效果对比

## 5. Data Pipeline Definition

训练样本定义为：

```text
left >> right
left  = reactants
right = product
```

训练时的数据走向：

1. 从 CSV 读一行
2. 拆成 `reactants` 和 `product`
3. `product` 送入 encoder tokenizer 和 encoder
4. `reactants` 送入 decoder tokenizer
5. 构造 decoder 输入与监督目标：
   `x = [BOS] + reactants[:-1]`
   `y = reactants + [EOS]`
6. 计算 `P(y_t | y_<t, product)`

关键约束：

- product 是条件，不进入 decoder token 序列
- reactants 是目标，不进入 encoder
- 不预测 `reagents`
- 评估时 reactants 侧需要固定标准化口径，否则 exact match 噪声会很大

## 6. Model Change List

为落地该方案，代码层面至少需要完成以下改造：

1. 修正 encoder tokenizer lowercasing 问题
2. 修正 encoder attention mask 未生效的问题
3. 在 decoder block 中加入 cross-attention 子层
4. 引入 aligner 模块
5. 定义新的训练 forward：
   输入 product ids、product mask、reactants ids、targets
6. 定义训练脚本：
   支持 Stage 1 / Stage 2 两阶段配置
7. 定义评估脚本：
   支持 greedy / beam search

代码改造优先级：

- 先打通最小可训练链路
- 再做配置化
- 最后再考虑训练效率优化

## 7. Validation Ladder

严格按下面顺序推进，不跳级：

1. 静态导入检查
2. tokenizer 行为检查
3. 一条样本前向 shape 检查
4. 一条样本 loss 反传检查
5. 1 batch 过拟合检查
6. 100 到 1000 条样本 smoke train
7. Stage 1 小规模训练
8. Stage 2 小规模训练
9. 全量训练
10. test 集正式评估

任何一级不稳定，都不要直接继续下一步。

## 8. Single-Person Execution Order

建议单人执行顺序：

### Day 1-2

- 清理 Stage 0 blockers
- 明确 tokenization 和标准化方案
- 产出最小数据管线说明

### Day 3-4

- 在 decoder 中加 cross-attention
- 加 aligner
- 跑通单条样本前向和反传

### Day 5-6

- 完成训练脚本
- 跑通 1 batch 过拟合
- 跑 100 到 1000 条样本 smoke train

### Day 7-9

- 正式跑 Stage 1
- 存 checkpoint
- 做中间评估和样本检查

### Day 10-12

- 跑 Stage 2
- 比较 Stage 1 / Stage 2 指标
- 调整解冻层数和学习率

### Day 13-14

- 在 test 集上出正式结果
- 做最小消融
- 形成方法总结和风险清单

## 9. High-Probability Decisions

为了提高成功率，优先做以下决策：

- 先做 cross-attention 方案，不先做复杂多路对齐模块
- encoder 全程冻结
- decoder 不一上来全量解冻
- 先过拟合小样本，再跑全量
- 先做可解释的 token-level 条件 memory，不先做单向量条件
- 评估时优先看 exact match 和有效分子率，不只看 loss

## 10. Main Risks

主要风险如下：

- encoder tokenizer/attention 细节如果不先修，训练结论会失真
- reactants 字符串顺序噪声会拉低 exact match
- decoder 如果从第一步就全量解冻，容易不稳定
- 条件接口如果太弱，模型会退化成无条件 SMILES 生成器
- full data 训练前如果没有 smoke train，排错成本会非常高

## 11. Success Criteria

本计划的成功标准不是“代码能跑”，而是同时满足：

- 训练链路稳定
- Stage 1 和 Stage 2 都能复现
- 模型对 product 有条件依赖
- test 集上拿到可解释的 top-k 结果
- 可以明确说明模型失败样本主要卡在哪类化学模式

## 12. Recommended Deliverables

最终应产出：

- 训练脚本
- 推理/评估脚本
- 配置文件
- 指标表
- 失败样本分析
- 简明方法说明文档

这份计划默认以“先做出可靠 baseline，再逐步增强”为原则，不追求第一版就做成最复杂结构。
