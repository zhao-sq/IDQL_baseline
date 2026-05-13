from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
import jax
import flax
from jaxrl5.networks.load_r3m import load_r3m_resnet18_into_flax

default_init = nn.initializers.xavier_uniform

def default_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(
        scale, mode="fan_avg", distribution="uniform"
    )

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not training)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x

class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 512
    activations: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate)(x, training=training)
            
        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x
    
class BasicBlock(nn.Module):
    features: int
    stride: int = 1
    norm_momentum: float = 0.9
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=default_init(),
            name="conv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=self.norm_momentum,
            epsilon=self.eps,
            name="bn1",
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=default_init(),
            name="conv2",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=self.norm_momentum,
            epsilon=self.eps,
            name="bn2",
        )(x)

        if residual.shape != x.shape:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                kernel_init=default_init(),
                name="downsample_conv",
            )(residual)
            residual = nn.BatchNorm(
                use_running_average=not training,
                momentum=self.norm_momentum,
                epsilon=self.eps,
                name="downsample_bn",
            )(residual)

        return nn.relu(residual + x)


class ResNet18Encoder(nn.Module):
    out_dim: int = 512
    norm_momentum: float = 0.9
    eps: float = 1e-5

    def _make_stage(self, x, features, blocks, stride, training, stage_name):
        x = BasicBlock(
            features=features,
            stride=stride,
            norm_momentum=self.norm_momentum,
            eps=self.eps,
            name=f"{stage_name}_0",
        )(x, training=training)
        for i in range(1, blocks):
            x = BasicBlock(
                features=features,
                stride=1,
                norm_momentum=self.norm_momentum,
                eps=self.eps,
                name=f"{stage_name}_{i}",
            )(x, training=training)
        return x

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # x: [B, H, W, C], 通常 C=3
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            kernel_init=default_init(),
            name="conv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=self.norm_momentum,
            epsilon=self.eps,
            name="bn1",
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        x = self._make_stage(x, 64, 2, 1, training, "layer1")
        x = self._make_stage(x, 128, 2, 2, training, "layer2")
        x = self._make_stage(x, 256, 2, 2, training, "layer3")
        x = self._make_stage(x, 512, 2, 2, training, "layer4")

        # global average pool -> [B, 512]
        x = jnp.mean(x, axis=(1, 2))

        # 默认 ResNet18 最终就是 512 维；如果你想再投影可以打开
        if self.out_dim != 512:
            x = nn.Dense(self.out_dim, kernel_init=default_init(), name="proj")(x)

        return x
    
class R3MResNet18MLPPolicy(nn.Module):
    action_dim: int
    actor_num_blocks: int
    actor_dropout_rate: Optional[float] = None
    actor_layer_norm: bool = False
    mlp_hidden_dim: int = 256
    visual_out_dim: int = 512
    activations: Callable = mish

    @nn.compact
    def __call__(self, image: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        feat = ResNet18Encoder(
            out_dim=self.visual_out_dim,
            name="visual_encoder",
        )(image, training=training)

        act = MLPResNet(
            num_blocks=self.actor_num_blocks,
            out_dim=self.action_dim,
            dropout_rate=self.actor_dropout_rate,
            use_layer_norm=self.actor_layer_norm,
            hidden_dim=self.mlp_hidden_dim,
            activations=self.activations,
            name="policy_head",
        )(feat, training=training)

        return act
    
# test the r3m model
def init_model(rng, image_shape, action_dim):
    model = R3MResNet18MLPPolicy(
        action_dim=action_dim,
        actor_num_blocks=3,
        actor_dropout_rate=0.1,
        actor_layer_norm=False,
        mlp_hidden_dim=256,
        visual_out_dim=512,
        activations=mish,
    )

    dummy = jnp.zeros(image_shape, dtype=jnp.float32)  # e.g. (1, 224, 224, 3)
    variables = model.init(
        {"params": rng, "dropout": rng},
        dummy,
        training=False,
    )
    return model, variables

def main():
    # =========================
    # 1. 配置
    # =========================
    rng = jax.random.PRNGKey(0)

    batch_size = 1
    image_shape = (batch_size, 224, 224, 3)   # NHWC
    action_dim = 7
    torch_ckpt_path = "/home/shuqi/.r3m/r3m_18/model.pt"  # 改成你的checkpoint路径

    # =========================
    # 2. 构建模型
    # =========================
    model = R3MResNet18MLPPolicy(
        action_dim=action_dim,
        actor_num_blocks=3,
        actor_dropout_rate=0.1,
        actor_layer_norm=False,
        mlp_hidden_dim=256,
        visual_out_dim=512,
        activations=mish,
    )

    # =========================
    # 3. 初始化 Flax variables
    # =========================
    dummy_image = jnp.zeros(image_shape, dtype=jnp.float32)

    variables = model.init(
        {"params": rng, "dropout": rng},
        dummy_image,
        training=False,
    )

    print("Model initialized.")
    print("Variable collections:", variables.keys())

    # =========================
    # 4. 加载 PyTorch R3M checkpoint
    #    只会覆盖 visual_encoder 部分
    # =========================
    variables = load_r3m_resnet18_into_flax(
        variables=variables,
        torch_ckpt_path=torch_ckpt_path,
    )

    print(f"Loaded R3M checkpoint from: {torch_ckpt_path}")

    # =========================
    # 5. 做一次前向测试
    # =========================
    actions = model.apply(
        variables,
        dummy_image,
        training=False,
    )

    print("Forward pass done.")
    print("Output action shape:", actions.shape)
    print("Output action:", actions)


if __name__ == "__main__":
    main()