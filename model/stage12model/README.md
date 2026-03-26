# stage12model

该目录存放旧版 Stage 1 / Stage 2 训练主干代码：

- `train_retrosynthesis.py`: Stage1/Stage2 训练主入口（旧版主干）
- `evaluate_checkpoint.py`: 对应 checkpoint 评估入口（旧版主干）

项目根目录下的 `model/train_retrosynthesis.py` 与 `model/evaluate_checkpoint.py` 现在为兼容包装器，会转发到本目录实现。
