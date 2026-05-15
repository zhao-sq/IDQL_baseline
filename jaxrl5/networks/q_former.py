from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl5.networks.mlp import MLP, default_init
from jaxrl5.networks.resnet import MLPResNet


def mish(x: jnp.ndarray) -> jnp.ndarray:
    """Apply Mish activation."""
    return x * jnp.tanh(nn.softplus(x))


class LearnableQueryTokens(nn.Module):
    """Return learnable query tokens, optionally indexed by a task id."""

    num_query_tokens: int
    dim: int
    num_tasks: int = 1
    init_std: float = 0.02

    @nn.compact
    def __call__(
        self,
        batch_size: int,
        task_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Build query tokens with shape [B, Tq, D]."""
        query_tokens = self.param(
            "query_tokens",
            nn.initializers.normal(self.init_std),
            (self.num_tasks, self.num_query_tokens, self.dim),
        )
        if task_ids is None:
            task_ids = jnp.zeros((batch_size,), dtype=jnp.int32)
        task_ids = task_ids.astype(jnp.int32)
        return query_tokens[task_ids]


class MultiHeadAttention(nn.Module):
    """Small multi-head attention block for query and token fusion."""

    embed_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key_value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """Attend from query tokens to key/value tokens."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        head_dim = self.embed_dim // self.num_heads
        batch_size = query.shape[0]

        q = nn.Dense(self.embed_dim, kernel_init=default_init())(query)
        k = nn.Dense(self.embed_dim, kernel_init=default_init())(key_value)
        v = nn.Dense(self.embed_dim, kernel_init=default_init())(key_value)

        def split_heads(x: jnp.ndarray) -> jnp.ndarray:
            """Reshape [B, T, D] into [B, H, T, Dh]."""
            x = x.reshape(batch_size, x.shape[1], self.num_heads, head_dim)
            return jnp.swapaxes(x, 1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(head_dim)
        if mask is not None:
            logits = jnp.where(mask[None, None, :, :], -1e10, logits)

        attn = nn.softmax(logits, axis=-1)
        if self.dropout_rate > 0:
            attn = nn.Dropout(rate=self.dropout_rate)(
                attn,
                deterministic=not training,
            )

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = jnp.swapaxes(out, 1, 2).reshape(batch_size, query.shape[1], self.embed_dim)
        return nn.Dense(self.embed_dim, kernel_init=default_init())(out)


class TransformerFeedForward(nn.Module):
    """Feed-forward block used after attention."""

    embed_dim: int
    ff_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply two dense layers with GELU activation."""
        x = nn.Dense(self.ff_dim, kernel_init=default_init())(x)
        x = nn.gelu(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.embed_dim, kernel_init=default_init())(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        return x


class CrossAttentionDecoderBlock(nn.Module):
    """One Q-former layer: query self-attention, cross-attention, and FFN."""

    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        query_tokens: jnp.ndarray,
        context_tokens: jnp.ndarray,
        query_mask: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """Update query tokens by attending to image patch tokens."""
        x = query_tokens
        x = x + MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name="self_attn",
        )(nn.LayerNorm()(x), nn.LayerNorm()(x), mask=query_mask, training=training)
        x = x + MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name="cross_attn",
        )(nn.LayerNorm()(x), context_tokens, training=training)
        x = x + TransformerFeedForward(
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
            name="ffn",
        )(nn.LayerNorm()(x), training=training)
        return x


class CrossAttentionPooling(nn.Module):
    """Pool patch tokens with externally supplied query tokens."""

    token_dim: int
    embed_dim: int
    out_dim: int
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 512
    dropout_rate: float = 0.1
    mask_type: str = "none"

    def _query_mask(self, num_queries: int) -> Optional[jnp.ndarray]:
        """Create the optional query-side self-attention mask."""
        if self.mask_type == "none":
            return None
        if self.mask_type == "query_separate":
            mask = jnp.ones((num_queries, num_queries), dtype=bool)
            return mask.at[jnp.diag_indices(num_queries)].set(False)
        raise ValueError(f"Unknown mask_type: {self.mask_type}")

    @nn.compact
    def __call__(
        self,
        patch_tokens: jnp.ndarray,
        query_tokens: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """Return pooled query features with shape [B, Tq, out_dim]."""
        tokens = nn.Dense(self.embed_dim, kernel_init=default_init())(patch_tokens)
        tokens = nn.LayerNorm()(tokens)
        query_mask = self._query_mask(query_tokens.shape[1])

        x = query_tokens
        for i in range(self.num_layers):
            x = CrossAttentionDecoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                name=f"decoder_{i}",
            )(x, tokens, query_mask=query_mask, training=training)

        x = nn.LayerNorm()(x)
        return nn.Dense(self.out_dim, kernel_init=default_init())(x)


class QFormerEncoder(nn.Module):
    """Convert DINO patch tokens into a flat global conditioning vector."""

    token_dim: int = 384
    pooled_dim: int = 64
    num_objects: int = 1
    num_containers: int = 1
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 512
    dropout_rate: float = 0.1
    mask_type: str = "none"
    query_init_std: float = 0.02

    @nn.compact
    def __call__(self, patch_tokens: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Pool [B, num_patches, token_dim] tokens using fixed task id [0, 0]."""
        if patch_tokens.ndim != 3:
            raise ValueError(f"QFormerEncoder expects [B, L, D], got {patch_tokens.shape}")
        if patch_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"QFormerEncoder expected token_dim={self.token_dim}, got {patch_tokens.shape[-1]}"
            )

        batch_size = patch_tokens.shape[0]
        task_ids = jnp.zeros((batch_size,), dtype=jnp.int32)
        obj_query = LearnableQueryTokens(
            num_query_tokens=1,
            dim=self.token_dim,
            num_tasks=self.num_objects,
            init_std=self.query_init_std,
            name="obj_query",
        )(batch_size, task_ids=task_ids)
        cont_query = LearnableQueryTokens(
            num_query_tokens=1,
            dim=self.token_dim,
            num_tasks=self.num_containers,
            init_std=self.query_init_std,
            name="cont_query",
        )(batch_size, task_ids=task_ids)
        query_tokens = jnp.concatenate([obj_query, cont_query], axis=1)
        pooled = CrossAttentionPooling(
            token_dim=self.token_dim,
            embed_dim=self.token_dim,
            out_dim=self.pooled_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
            mask_type=self.mask_type,
            name="fusion",
        )(patch_tokens, query_tokens, training=training)
        return pooled.reshape(batch_size, -1)


class Conv1DBlock(nn.Module):
    """Conv1D -> GroupNorm -> Mish block."""

    features: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply a channel-last 1D convolution block."""
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            kernel_init=default_init(),
        )(x)
        x = nn.GroupNorm(num_groups=self.n_groups)(x)
        return mish(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual 1D convolution block with FiLM conditioning."""

    out_channels: int
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """Apply FiLM modulation from the conditioning vector."""
        residual = x
        out = Conv1DBlock(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
        )(x)

        cond_channels = self.out_channels * 2 if self.use_film_scale_modulation else self.out_channels
        cond_embed = nn.Dense(cond_channels, kernel_init=default_init())(mish(cond))
        cond_embed = cond_embed[:, None, :]
        if self.use_film_scale_modulation:
            scale, bias = jnp.split(cond_embed, 2, axis=-1)
            out = scale * out + bias
        else:
            out = out + cond_embed

        out = Conv1DBlock(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
        )(out)
        if residual.shape[-1] != self.out_channels:
            residual = nn.Dense(self.out_channels, kernel_init=default_init())(residual)
        return out + residual


class FiLMConv1DUNet(nn.Module):
    """FiLM-conditioned U-Net style head for vector actions."""

    action_dim: int
    down_dims: Sequence[int] = (128, 256, 512)
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        time_cond: jnp.ndarray,
        global_cond: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict DDPM noise from actions, timestep features, and global condition."""
        cond = jnp.concatenate([time_cond, global_cond], axis=-1)
        x = actions[:, None, :]
        in_out = [(self.action_dim, self.down_dims[0])] + list(
            zip(self.down_dims[:-1], self.down_dims[1:])
        )

        skip_features = []
        for _, dim_out in in_out:
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                use_film_scale_modulation=self.use_film_scale_modulation,
            )(x, cond)
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                use_film_scale_modulation=self.use_film_scale_modulation,
            )(x, cond)
            skip_features.append(x)

        for _ in range(2):
            x = ConditionalResidualBlock1D(
                out_channels=self.down_dims[-1],
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                use_film_scale_modulation=self.use_film_scale_modulation,
            )(x, cond)

        for dim_out, _ in reversed(in_out[1:]):
            x = jnp.concatenate([x, skip_features.pop()], axis=-1)
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                use_film_scale_modulation=self.use_film_scale_modulation,
            )(x, cond)
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                use_film_scale_modulation=self.use_film_scale_modulation,
            )(x, cond)

        x = Conv1DBlock(
            features=self.down_dims[0],
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
        )(x)
        x = nn.Conv(
            features=self.action_dim,
            kernel_size=(1,),
            kernel_init=default_init(),
        )(x)
        return jnp.squeeze(x, axis=1)


class QFormerUNetBase(nn.Module):
    """Base model: Q-former global condition followed by a FiLM U-Net head."""

    action_dim: int
    token_dim: int = 384
    pooled_dim: int = 64
    num_objects: int = 1
    num_containers: int = 1
    fusion_num_layers: int = 4
    fusion_num_heads: int = 8
    fusion_ff_dim: int = 512
    fusion_dropout: float = 0.1
    fusion_mask_type: str = "none"
    down_dims: Sequence[int] = (128, 256, 512)
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        time_cond: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """Predict DDPM noise from patch-token observations and timestep features."""
        global_cond = QFormerEncoder(
            token_dim=self.token_dim,
            pooled_dim=self.pooled_dim,
            num_objects=self.num_objects,
            num_containers=self.num_containers,
            num_layers=self.fusion_num_layers,
            num_heads=self.fusion_num_heads,
            ff_dim=self.fusion_ff_dim,
            dropout_rate=self.fusion_dropout,
            mask_type=self.fusion_mask_type,
            name="q_former",
        )(observations, training=training)
        return FiLMConv1DUNet(
            action_dim=self.action_dim,
            down_dims=self.down_dims,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            use_film_scale_modulation=self.use_film_scale_modulation,
            name="policy_head",
        )(actions, time_cond, global_cond)


class QFormerMLPBase(nn.Module):
    """Base model: Q-former global condition followed by an MLP head."""

    action_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    token_dim: int = 384
    pooled_dim: int = 64
    num_objects: int = 1
    num_containers: int = 1
    fusion_num_layers: int = 4
    fusion_num_heads: int = 8
    fusion_ff_dim: int = 512
    fusion_dropout: float = 0.1
    fusion_mask_type: str = "none"
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        time_cond: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """Predict DDPM noise with a Q-former encoder and MLP policy head."""
        global_cond = QFormerEncoder(
            token_dim=self.token_dim,
            pooled_dim=self.pooled_dim,
            num_objects=self.num_objects,
            num_containers=self.num_containers,
            num_layers=self.fusion_num_layers,
            num_heads=self.fusion_num_heads,
            ff_dim=self.fusion_ff_dim,
            dropout_rate=self.fusion_dropout,
            mask_type=self.fusion_mask_type,
            name="q_former",
        )(observations, training=training)
        inputs = jnp.concatenate([actions, global_cond, time_cond], axis=-1)
        return MLP(
            hidden_dims=tuple(list(self.hidden_dims) + [self.action_dim]),
            activations=mish,
            activate_final=False,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            name="policy_head",
        )(inputs, training=training)


class QFormerMLPResNetBase(nn.Module):
    """Base model: Q-former global condition followed by an MLPResNet head."""

    action_dim: int
    num_blocks: int = 2
    hidden_dim: int = 512
    token_dim: int = 384
    pooled_dim: int = 64
    num_objects: int = 1
    num_containers: int = 1
    fusion_num_layers: int = 4
    fusion_num_heads: int = 8
    fusion_ff_dim: int = 512
    fusion_dropout: float = 0.1
    fusion_mask_type: str = "none"
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        time_cond: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """Predict DDPM noise with a Q-former encoder and MLPResNet policy head."""
        global_cond = QFormerEncoder(
            token_dim=self.token_dim,
            pooled_dim=self.pooled_dim,
            num_objects=self.num_objects,
            num_containers=self.num_containers,
            num_layers=self.fusion_num_layers,
            num_heads=self.fusion_num_heads,
            ff_dim=self.fusion_ff_dim,
            dropout_rate=self.fusion_dropout,
            mask_type=self.fusion_mask_type,
            name="q_former",
        )(observations, training=training)
        inputs = jnp.concatenate([actions, global_cond, time_cond], axis=-1)
        return MLPResNet(
            num_blocks=self.num_blocks,
            out_dim=self.action_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm,
            hidden_dim=self.hidden_dim,
            activations=mish,
            name="policy_head",
        )(inputs, training=training)


class QFormerDDPM(nn.Module):
    """DDPM wrapper that keeps Q-former as the reverse base model."""

    reverse_encoder_cls: Callable[..., nn.Module]
    time_preprocess_cls: Callable[..., nn.Module]
    cond_encoder_cls: Callable[..., nn.Module]

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        time: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """Preprocess timestep and call the Q-former reverse base model."""
        time_features = self.time_preprocess_cls()(time)
        time_cond = self.cond_encoder_cls()(time_features, training=training)
        return self.reverse_encoder_cls()(observations, actions, time_cond, training=training)


class QFormerStateActionValue(nn.Module):
    """Critic: independent Q-former global condition followed by an MLP Q head."""

    base_cls: Callable[..., nn.Module]
    token_dim: int = 384
    pooled_dim: int = 64
    num_objects: int = 1
    num_containers: int = 1
    fusion_num_layers: int = 4
    fusion_num_heads: int = 8
    fusion_ff_dim: int = 512
    fusion_dropout: float = 0.1
    fusion_mask_type: str = "none"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Evaluate Q(global_cond, action)."""
        global_cond = QFormerEncoder(
            token_dim=self.token_dim,
            pooled_dim=self.pooled_dim,
            num_objects=self.num_objects,
            num_containers=self.num_containers,
            num_layers=self.fusion_num_layers,
            num_heads=self.fusion_num_heads,
            ff_dim=self.fusion_ff_dim,
            dropout_rate=self.fusion_dropout,
            mask_type=self.fusion_mask_type,
            name="q_former",
        )(observations, training=False)
        inputs = jnp.concatenate([global_cond, actions], axis=-1)
        outputs = self.base_cls()(inputs)
        value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


class QFormerStateValue(nn.Module):
    """Value network: independent Q-former global condition followed by an MLP."""

    base_cls: Callable[..., nn.Module]
    token_dim: int = 384
    pooled_dim: int = 64
    num_objects: int = 1
    num_containers: int = 1
    fusion_num_layers: int = 4
    fusion_num_heads: int = 8
    fusion_ff_dim: int = 512
    fusion_dropout: float = 0.1
    fusion_mask_type: str = "none"

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Evaluate V(global_cond)."""
        global_cond = QFormerEncoder(
            token_dim=self.token_dim,
            pooled_dim=self.pooled_dim,
            num_objects=self.num_objects,
            num_containers=self.num_containers,
            num_layers=self.fusion_num_layers,
            num_heads=self.fusion_num_heads,
            ff_dim=self.fusion_ff_dim,
            dropout_rate=self.fusion_dropout,
            mask_type=self.fusion_mask_type,
            name="q_former",
        )(observations, training=False)
        outputs = self.base_cls()(global_cond)
        value = nn.Dense(1, kernel_init=default_init(), name="OutputVDense")(outputs)
        return jnp.squeeze(value, -1)
