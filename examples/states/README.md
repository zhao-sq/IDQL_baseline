## Online

### SAC
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --config=configs/sac_config.py \
                --notqdm
```
### DroQ
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 300000 \
                --config=configs/droq_config.py \
                --notqdm
```
### RedQ
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 300000 \
                --config=configs/redq_config.py \
                --notqdm
```

## Offline

###
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_offline.py --env_name=halfcheetah-expert-v2 \
                --config=configs/bc_config.py
```

## Q_former DDPM IQL

`train_diffusion_offline.py` 在自定义数据集任务下会根据 architecture 选择 observation space：

- 普通分支：使用 `(obs_dim,)`
- `Q_former` 分支：使用完整 token shape，例如 `(256, 384)`

用法：

```python
variant["rl_config"]["actor_architecture"] = "Q_former"
variant["rl_config"]["q_former_actor_head"] = "unet"  # 可选: "unet", "ln_resnet", "mlp"
variant["rl_config"]["critic_q_former_num_layers"] = 2
variant["rl_config"]["critic_q_former_num_heads"] = 4
```

`actor_architecture` 也可以直接写成 `Q_former+unet`、`Q_former+ln_resnet` 或 `Q_former+mlp`。不写 head 时默认是 `Q_former+unet`。

critic 的 Q-former 默认使用更小配置，减少 double-Q ensemble 带来的参数量；actor 的 Q-former 配置不受影响。

如果没有显式指定 `critic_architecture` 和 `value_architecture`，learner 会在 actor 为 `Q_former` 时自动让它们跟随为 `Q_former`。
