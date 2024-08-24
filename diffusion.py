import torch
from torch import nn
from data import default_skeleton, hierarchical_order
from transforms3d.euler import euler2mat
from typing import Optional
from enum import Enum
import torch.nn.functional as F
from dataclasses import dataclass
from accelerate import Accelerator
from rotation_conversions import euler_angles_to_matrix, matrix_to_axis_angle, quaternion_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d


class RotationType(Enum):
    MATRIX = "matrix"
    QUAT = "quat"
    ZHOU_6D = "zhou_6d"


def rotation_type_to_dim(rotation_type: RotationType) -> int:
    if rotation_type == RotationType.MATRIX:
        return 9
    elif rotation_type == RotationType.ZHOU_6D:
        return 6
    elif rotation_type == RotationType.QUAT:
        return 4


@dataclass
class Config:
    num_joints: int = len(hierarchical_order)
    rotation_type: RotationType = RotationType.ZHOU_6D
    reconstruction_loss_weight: float = 1.0
    context_loss_weight: float = 1.0


config = Config()


class ForwardKinematics(nn.Module):
    def __init__(self):
        super().__init__()
        self.bones = nn.ModuleDict({
            name: Bone(bone.axis, bone.direction, bone.length)
            for (name, bone) in default_skeleton.bone_map.items()
            if not name == "root"
        })

    def get_matrix(self, x: torch.Tensor, i: int) -> torch.Tensor:
        if config.rotation_type == RotationType.MATRIX:
            return x[:, i:i + 9].view(-1, 3, 3)
        elif config.rotation_type == RotationType.ZHOU_6D:
            return rotation_6d_to_matrix(x[:, i:i + 6])
        elif config.rotation_type == RotationType.QUAT:
            return quaternion_to_matrix(x[:, i:i + 4])
        else:
            raise ValueError(f"Invalid rotation type {config.rotation_type}")

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        rotation_dim = rotation_type_to_dim(config.rotation_type)
        # (B, (3 + rotation_dim * num_joints)) -> (B, 3 * num_joints)
        bone_cache = {}

        i = 0
        for bone in hierarchical_order:
            if bone == "root":
                tail_position = x[:, i:i+3]
                i += 3
                global_rotation = self.get_matrix(x, i)
                i += rotation_dim
                bone_cache[bone] = (global_rotation, tail_position)
            else:
                parent = default_skeleton.bone_map[bone].parent
                assert parent is not None
                assert parent.name in bone_cache
                local_rotation = self.get_matrix(x, i)
                i += rotation_dim
                bone_cache[bone] = self.bones[bone](
                    *bone_cache[parent.name], local_rotation)

        x = torch.cat([bone_cache[bone][1]
                      for bone in hierarchical_order], dim=1)

        x = x.reshape(*shape[:-1], x.size(-1))
        return x


class Bone(nn.Module):
    def __init__(self, axis, direction, length):
        super().__init__()
        axis = torch.tensor(axis).deg2rad().to(torch.float32)
        bind = torch.tensor(euler2mat(*axis.tolist())).to(torch.float32)
        inverse_bind = bind.inverse().to(torch.float32)
        direction = torch.tensor(direction).to(torch.float32)
        length = torch.tensor(length).to(torch.float32)
        self.register_buffer("bind", bind)
        self.register_buffer("inverse_bind", inverse_bind)
        self.register_buffer("direction", direction)
        self.register_buffer("length", length)

    def forward(self, parent_global_transform, parent_tail_position, local_rotation):
        global_transform = parent_global_transform @ self.bind @ local_rotation @ self.inverse_bind
        tail_position = parent_tail_position + self.length * \
            (global_transform @ self.direction)
        return global_transform, tail_position


class FeedFowardBlock(nn.Module):
    def __init__(self, input_embedding_size: int, hidden_embedding_size: int, output_embedding_size: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_embedding_size, hidden_embedding_size),
            nn.GELU(),
            nn.Linear(hidden_embedding_size, output_embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        input_embedding_size: int,
        hidden_embedding_size: int,
        output_embedding_size: int,
        attention_head_count: int,
        dropout: float,
        device='cpu'
    ):
        super().__init__()
        self.device = device

        self.ln_1 = nn.LayerNorm(input_embedding_size)
        self.ln_2 = nn.LayerNorm(input_embedding_size)

        self.mlp = FeedFowardBlock(
            input_embedding_size, hidden_embedding_size, output_embedding_size, dropout)

        self.attn = nn.MultiheadAttention(
            input_embedding_size, attention_head_count, dropout, batch_first=True)

    def forward(self, query: torch.Tensor, key_value: Optional[torch.Tensor] = None, mask=True) -> torch.Tensor:
        length = query.size(1)

        attn_mask = torch.triu(torch.ones(length, length) *
                               float('-inf'), diagonal=1).to(self.device) if mask else None

        key_value = key_value if key_value is not None else query

        x, _ = self.attn(self.ln_1(query), self.ln_1(key_value),
                         self.ln_1(key_value), attn_mask=attn_mask)

        x = self.mlp(self.ln_2(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        number_of_layers: int,
        input_embedding_size: int,
        hidden_embedding_size: int,
        output_embedding_size: int,
        attention_head_count: int,
        dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(input_embedding_size,
                                                    hidden_embedding_size,
                                                    input_embedding_size,
                                                    attention_head_count,
                                                    dropout) for _ in range(number_of_layers)])

        self.ln = nn.LayerNorm(input_embedding_size)
        self.projection = FeedFowardBlock(
            input_embedding_size, hidden_embedding_size, output_embedding_size, dropout)

    def forward(self, x: torch.Tensor, encoder_input: Optional[torch.Tensor] = None, mask=True) -> torch.Tensor:
        x_ = x
        for layer in self.layers:
            x_ = layer(x_, key_value=encoder_input, mask=mask)

        x_ = self.ln(x_)
        x_ = x + x_
        x_ = self.projection(x_)
        return x_


class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        layers: int,
        input_embedding_size: int,
        hidden_embedding_size: int,
        output_embedding_size: int,
        attention_head_count: int,
        dropout: float
    ):
        super().__init__()

        self.encoder = Transformer(layers, input_embedding_size, hidden_embedding_size,
                                   output_embedding_size, attention_head_count, dropout)
        self.decoder = Transformer(layers, input_embedding_size, hidden_embedding_size,
                                   output_embedding_size, attention_head_count, dropout)
        self.cross_attention = Transformer(
            layers, input_embedding_size, hidden_embedding_size, output_embedding_size, attention_head_count, dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.encoder(c, mask=False)
        x = self.decoder(x, mask=True)
        x = self.cross_attention(x, c, mask=False)
        return x


class Denoiser(nn.Module):
    def __init__(
        self,
        sequential: bool = True,
        encoder_layers: int = 12,
        decoder_layers: int = 12,
        cross_attention_layers: int = 12,
        attention_head_count: int = 8,
        input_embedding_size: int = 176,
        block_size: int = 60,
        feature_size: int = 3 +
            rotation_type_to_dim(config.rotation_type) * config.num_joints,
        timesteps: int = 300,
        dropout: float = 0.1
    ):
        super().__init__()

        self.sequential = sequential

        if self.sequential:
            self.encoder = Transformer(encoder_layers, input_embedding_size,
                                       input_embedding_size * 2, input_embedding_size, attention_head_count, dropout)
            self.decoder = Transformer(decoder_layers, input_embedding_size,
                                       input_embedding_size * 2, input_embedding_size, attention_head_count, dropout)
            self.cross_attention = Transformer(cross_attention_layers, input_embedding_size,
                                               input_embedding_size * 2, input_embedding_size, attention_head_count, dropout)
        else:
            self.blocks = nn.ModuleList(
                [CrossAttentionTransformer(1, input_embedding_size, input_embedding_size * 2, input_embedding_size,
                                           attention_head_count, dropout) for _ in range(cross_attention_layers)]
            )

        self.positional_embedding = nn.Embedding(
            block_size, input_embedding_size)
        self.time_embedding = nn.Embedding(timesteps, input_embedding_size)
        self.feature_embedding = nn.Linear(feature_size, input_embedding_size)
        self.output = nn.Linear(input_embedding_size, feature_size)

        self.apply(self._init_weights)

     # from andrej karpathy gpt-2 code
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, c: torch.Tensor,  c_i: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x = (b, block_size, feature_length)  c_i = (b, context_length,)   t = (b,)

        time_embd = self.time_embedding(t).unsqueeze(1)

        x_feature_embd = self.feature_embedding(x)
        x_position_embd = self.positional_embedding(torch.arange(x.size(1)))

        c_feature_embd = self.feature_embedding(c)
        c_position_embd = self.positional_embedding(c_i)

        x = x_feature_embd + x_position_embd + time_embd
        c = c_feature_embd + c_position_embd + time_embd

        if self.sequential:
            c = self.encoder(c, mask=False)
            x = self.decoder(x, mask=True)
            x = self.cross_attention(x, c, mask=False)
        else:
            for block in self.blocks:
                x = block(x, c)

        x = self.output(x)
        return x


def forward_kinematics(x: torch.Tensor) -> torch.Tensor:
    if not hasattr(forward_kinematics, 'block'):
        forward_kinematics.block = ForwardKinematics()
    return forward_kinematics.block(x)


class Diffusion(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()
        betas = self.linear_beta_schedule(timesteps)
        alphas = 1. - betas

        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1. - alphas_cumprod))

        self.denoiser = Denoiser(sequential=False)

    def forward_diffusion_sample(self, x_0, t) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_0)
        return self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise, noise

    def linear_beta_schedule(self, timesteps: int):
        scale = 1000 / timesteps
        start = scale * 0.0001
        end = scale * 0.02
        return torch.linspace(start, end, timesteps)

    def extract(self, a, t, x_shape):
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1, ) * (len(x_shape) - 1)))

    def compute_loss(self, x: torch.Tensor, c_i: torch.Tensor, t: torch.Tensor):
        '''
        Loss components
            1. Global loss between predicted clean motion and ground truth
            2. Reconstruction loss between predicted clean motion and noisy motion
            3. Reconstruction loss between context keypoints and the same keypoints in the predicted clean motion
        '''
        c = x.gather(-2, c_i.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        x_t, _ = self.forward_diffusion_sample(x, t)
        # predicted clean motion at time t
        x_hat_t = self.denoiser(x_t, c, c_i, t)

        # Forward kinematics
        x_f = forward_kinematics(x)
        x_hat_f = forward_kinematics(x_hat_t)

        # Global loss
        l_g = F.mse_loss(x_hat_t, x).sqrt()

        # Reconstruction loss
        l_r = F.mse_loss(x_hat_f, x_f).sqrt()

        # Context loss
        l_c = F.mse_loss(
            x_hat_f.gather(-2, c_i.unsqueeze(-1).expand(-1, -
                           1, x_hat_f.size(-1))),
            x_f.gather(-2, c_i.unsqueeze(-1).expand(-1, -1, x_f.size(-1)))
        ).sqrt()

        return l_g + l_r * config.reconstruction_loss_weight + l_c * config.context_loss_weight


class Dataset:
    def __init__(self):
        ...

    @staticmethod
    def from_cmu() -> 'Dataset':
        ...

    @staticmethod
    def from_kit_ml() -> 'Dataset':
        ...


if __name__ == "__main__":
    diffusion = Diffusion(400)

    count = 0
    for p in diffusion.parameters():
        count += p.numel()
    print(f"{count * 4 / 1024 / 1024:.2f} Mb")

    c_i = torch.randint(0, 60, (64, 10,))
    x = torch.randn(64, 60, 3 + config.num_joints *
                    rotation_type_to_dim(config.rotation_type))
    t = torch.randint(0, 300, (64,))

    print(diffusion.compute_loss(x, c_i, t))
