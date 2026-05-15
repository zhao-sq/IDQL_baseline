# launcher/examples

## 概述

这里放实验启动脚本，负责组装 `variant`、设置 sweep 参数，并调用 `examples.states.train_diffusion_offline.call_main`。

`train_ddpm_iql_finetune.py` 当前使用 `DDPMIQLLearner`。当 `actor_architecture="Q_former"` 时，默认 actor head 是 `unet`，也可以切换为 `ln_resnet` 或 `mlp`。

## 依赖

- `absl`
- `numpy`
- `examples.states.train_diffusion_offline`
- `launcher.hyperparameters`

## 用法

```python
rl_config=dict(
    model_cls="DDPMIQLLearner",
    actor_architecture="Q_former",
    q_former_actor_head="unet",  # "unet", "ln_resnet", "mlp"
    critic_q_former_pooled_dim=32,
    critic_q_former_num_layers=2,
    critic_q_former_num_heads=4,
    critic_q_former_ff_dim=256,
)
```

critic 的 Q-former 使用单独轻量配置，actor 的 Q-former 仍然由 `q_former_*` 参数控制。

也可以直接写：

```python
actor_architecture="Q_former+ln_resnet"
actor_architecture="Q_former+mlp"
```

## 示例

```bash
python launcher/examples/train_ddpm_iql_finetune.py --variant=0
```
