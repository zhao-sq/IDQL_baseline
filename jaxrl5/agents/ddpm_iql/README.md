# DDPM IQL Learner

## 概述

`ddpm_iql_learner.py` 实现 DDPM actor + IQL critic/value。`Q_former` 分支支持三种 actor head：

- actor: `Q_former + FiLM U-Net`，默认
- actor: `Q_former + MLPResNet`
- actor: `Q_former + MLP`
- critic: `Q_former + MLP`
- value: `Q_former + MLP`

critic 的 Q-former 有单独的轻量配置，默认比 actor 小：

- `critic_q_former_pooled_dim=32`
- `critic_q_former_num_layers=2`
- `critic_q_former_num_heads=4`
- `critic_q_former_ff_dim=256`

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
    q_former_actor_head="unet",  # 可选: "unet", "ln_resnet", "mlp"
    critic_q_former_pooled_dim=32,
    critic_q_former_num_layers=2,
    critic_q_former_num_heads=4,
    critic_q_former_ff_dim=256,
    critic_architecture="Q_former",
    value_architecture="Q_former",
)
```

如果不显式传 `critic_architecture` / `value_architecture`，当 actor 是 `Q_former` 时它们会默认跟随为 `Q_former`。

也可以把 head 写在 `actor_architecture` 里：

```python
actor_architecture="Q_former+ln_resnet"
actor_architecture="Q_former+mlp"
```
