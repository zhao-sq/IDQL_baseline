# jaxrl5.data

## 概述

这里放离线数据集和 replay buffer。`CustomDataset` 现在提供两个 observation shape helper：

- `get_obs_dim()`：返回最后一维，兼容原来的一维 observation 网络。
- `get_obs_shape()`：返回完整 per-sample shape，供 `Q_former` 使用，例如 `(256, 384)`。
- `sample_jax_batch()`：先在 CPU 侧采样一个 batch，再只把该 batch 放到 JAX 设备上，适合 `Q_former` 这类大 token 数据。

## 依赖

- `numpy`
- `gym`
- `jaxrl5.data.dataset.Dataset`

## 用法

```python
ds = CustomDataset("pick_and_lift")
obs_shape = ds.get_obs_shape()
action_dim = ds.get_action_dim()
batch = ds.sample_jax_batch(512)
```

## 示例

```python
observation_space = gym.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=ds.get_obs_shape(),
    dtype=np.float32,
)
```
