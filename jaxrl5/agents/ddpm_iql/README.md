# DDPM IQL Learner

## 概述

`ddpm_iql_learner.py` 实现 DDPM actor + IQL critic/value。新增 `Q_former` 分支后：

- actor: `Q_former + FiLM U-Net`
- critic: `Q_former + MLP`
- value: `Q_former + MLP`

只有在 architecture 设置为 `Q_former` 时才走 token observation 逻辑，其他 actor/critic/value 分支保持原来的 MLP 或 ResNet 初始化方式。

## 依赖

- `jax`
- `flax`
- `optax`
- `gym`
- `jaxrl5.networks`

## API/用法

```python
agent = DDPMIQLLearner.create(
    seed=0,
    observation_space=observation_space,  # shape=(256, 384)
    action_space=action_space,
    actor_architecture="Q_former",
    critic_architecture="Q_former",
    value_architecture="Q_former",
)
```

如果不显式传 `critic_architecture` / `value_architecture`，当 actor 是 `Q_former` 时它们会默认跟随为 `Q_former`。
