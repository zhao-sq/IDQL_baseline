# jaxrl5.networks

## 概述

这里放 JAX/Flax 网络模块。`q_former.py` 新增了面向 DINO patch token 的 Q-former 条件编码：

- 输入 observation: `[B, 256, 384]`
- 固定 task id: `[0, 0]`
- 输出 global condition: `[B, 2 * pooled_dim]`
- actor 默认使用 `QFormerDDPM + QFormerUNetBase`
- actor 也可以使用 `QFormerDDPM + QFormerMLPResNetBase`
- actor 也可以使用 `QFormerDDPM + QFormerMLPBase`
- critic/value 使用各自独立的 `QFormerStateActionValue` / `QFormerStateValue`

actor、critic、value 之间不共享 Q-former 参数。

## 依赖

- `jax`
- `jax.numpy`
- `flax.linen`

## 用法

```python
from jaxrl5.networks import QFormerDDPM, QFormerUNetBase

# 通常不直接手动初始化，而是在 DDPMIQLLearner.create(...)
# 中设置 actor_architecture="Q_former"。
```

## 示例

```python
agent = DDPMIQLLearner.create(
    seed,
    observation_space,  # shape=(256, 384)
    action_space,
    actor_architecture="Q_former",
    q_former_actor_head="unet",  # "unet", "ln_resnet", "mlp"
)
```

如果想直接在 architecture 里指定 head，也可以写：

```python
actor_architecture="Q_former+ln_resnet"
actor_architecture="Q_former+mlp"
```
