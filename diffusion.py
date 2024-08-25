import torch
from torch import nn
from data import default_skeleton, hierarchical_order
from typing import Optional
from enum import Enum
import torch.nn.functional as F
from dataclasses import dataclass
from accelerate import Accelerator
from rotation_conversions import euler_angles_to_matrix, matrix_to_axis_angle, quaternion_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import math
from argparse import ArgumentParser, ArgumentTypeError
from utils.dataset_gen import generate_dataset


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
    block_size: int = 60
    batch_size: int = 64
    timesteps: int = 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        batch_dim = x.shape[:-1]
        x = x.reshape(-1, x.size(-1))

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

        x = x.reshape(*batch_dim, x.size(-1))
        return x


class Bone(nn.Module):
    def __init__(self, axis, direction, length):
        super().__init__()
        axis = torch.tensor(axis).deg2rad().to(torch.float32)
        bind = euler_angles_to_matrix(axis, 'XYZ')
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
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(input_embedding_size)
        self.ln_2 = nn.LayerNorm(input_embedding_size)

        self.mlp = FeedFowardBlock(
            input_embedding_size, hidden_embedding_size, output_embedding_size, dropout)

        self.attn = nn.MultiheadAttention(
            input_embedding_size, attention_head_count, dropout, batch_first=True)

    def forward(self, query: torch.Tensor, key_value: Optional[torch.Tensor] = None, mask=True) -> torch.Tensor:
        length = query.size(1)

        attn_mask = torch.triu(torch.ones(length, length, device=query.device) *
                               float('-inf'), diagonal=1) if mask else None

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
        input_embedding_size: int = 256,
        feature_size: int = 3 +
            rotation_type_to_dim(config.rotation_type) * config.num_joints,
        timesteps: int = config.timesteps,
        dropout: float = 0.1
    ):
        super().__init__()

        self.sequential = sequential

        hidden_embedding_size = input_embedding_size
        output_embedding_size = input_embedding_size

        if self.sequential:
            self.encoder = Transformer(encoder_layers, input_embedding_size,
                                       hidden_embedding_size, output_embedding_size, attention_head_count, dropout)
            self.decoder = Transformer(decoder_layers, input_embedding_size,
                                       hidden_embedding_size, output_embedding_size, attention_head_count, dropout)
            self.cross_attention = Transformer(cross_attention_layers, input_embedding_size,
                                               hidden_embedding_size, output_embedding_size, attention_head_count, dropout)
        else:
            self.blocks = nn.ModuleList(
                [CrossAttentionTransformer(1, input_embedding_size, input_embedding_size * 2, input_embedding_size,
                                           attention_head_count, dropout) for _ in range(cross_attention_layers)]
            )

        self.positional_embedding = nn.Embedding(
            config.block_size, input_embedding_size)
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
        x_position_embd = self.positional_embedding(
            torch.arange(x.size(1), device=x.device))

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
    return forward_kinematics.block.to(x.device)(x)


# https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) /
                      tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        betas = cosine_beta_schedule(config.timesteps).to(torch.float32)
        alphas = 1. - betas

        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance)

        self.denoiser = Denoiser(sequential=False)

    def forward_diffusion_sample(self, x_0, t) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_0)
        return self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise, noise

    def extract(self, a, t, x_shape):
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1, ) * (len(x_shape) - 1)))

    def compute_loss(self, x: torch.Tensor, c_i: torch.Tensor, t: torch.Tensor, p_mean: torch.Tensor, p_std: torch.Tensor):
        '''
        Loss components
            1. Global loss between predicted clean motion and ground truth
            2. Reconstruction loss of joint positions between predicted clean motion and noisy motion
            3. Reconstruction loss of joint positions between context keypoints and the same keypoints in the predicted clean motion
        '''
        c = x.gather(-2, c_i.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        x_t, _ = self.forward_diffusion_sample(x, t)
        # predicted clean motion at time t
        x_hat_t = self.denoiser(x_t, c, c_i, t)

        x = x.clone()

        x[:, :, :3] = x[:, :, : 3] * p_std + p_mean
        x_hat_t[:, :, :3] = x_hat_t[:, :, : 3] * p_std + p_mean

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

    @torch.no_grad()
    def sample(self, c: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        batch_size = c.size(0)

        motion = torch.randn(batch_size, config.block_size,
                             c.size(-1), device=c.device)

        for i in range(config.timesteps)[::-1]:
            t = torch.full((batch_size,), i, device=c.device)
            motion = self.sample_timestep(motion, c, c_i, t)
            # motion = motion.clamp(-1.0, 1.0)
        return motion

    @torch.no_grad()
    def sample_timestep(self, x_t: torch.Tensor, c: torch.Tensor, c_i: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        clean_motion = self.denoiser(x_t, c, c_i, t)
        posterior_variance = self.extract(
            self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t, device=x_t.device) * \
            posterior_variance.sqrt()

        return clean_motion + noise * (t != 0).to(torch.float32).reshape(-1, 1, 1)

    def size(self):
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count * 4 / 1024 / 1024  # MB


class PositionScalar(nn.Module):
    def __init__(self):
        ...

    def forward(self):
        ...


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            dataset_path: str,
            checkpoint_path: str,
            lr: float = 3e-4,
            epochs: int = 5000,
            batch_size: int = 64,
            train_test_val_ratio=[0.8, 0.1, 0.1],
            early_stopper_patience=5,
    ):
        torch.manual_seed(22)

        self.epochs = epochs
        self.epoch = 0
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.early_stopper = EarlyStopping(patience=early_stopper_patience)

        optimizer = torch.optim.AdamW(model.parameters(), lr)

        data = torch.load(dataset_path, map_location="cpu")
        self.mean = data['mean']
        self.std = data['std']
        self.rotation_type = data['rotation_type']
        dataset = TensorDataset(data['data'])

        train_dataset, test_dataset, val_dataset = random_split(
            dataset, train_test_val_ratio)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        accelerator = Accelerator()

        self.model, self.optimizer, self.train_loader, self.test_loader, self.val_loader = accelerator.prepare(
            model, optimizer, train_loader, test_loader, val_loader)
        self.accelerator = accelerator

        self.checkpoint_path = checkpoint_path

        if os.path.exists(self.checkpoint_path) and os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}:")
            self.load_checkpoint(self.checkpoint_path)
            self.log_stat()

    def log_stat(self):
        print(f"Epoch {self.epoch}: train loss - {self.train_loss}, val loss - {self.val_loss}")

    @torch.no_grad()
    def evaluate_loss(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        for batch in data_loader:
            x = batch[0]
            c_length = int(torch.randint(1, config.block_size // 3, ()).item())
            c_i = torch.randint(0, config.block_size,
                                (x.size(0), c_length,), device=x.device)
            t = torch.randint(0, config.timesteps,
                              (x.size(0),), device=x.device)
            loss = self.model.compute_loss(
                x, c_i, t, self.mean.to(x.device), self.std.to(x.device))
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(data_loader)

    def train(self):
        self.model.train()
        for _ in range(self.epoch, self.epochs):
            self.epoch += 1
            total_loss = 0

            for batch in self.train_loader:
                x = batch[0]

                c_length = int(torch.randint(
                    1, config.block_size // 2, (),).item())
                print(c_length)
                c_i = torch.randint(0, config.block_size,
                                    (x.size(0), c_length,), device=x.device)
                t = torch.randint(0, config.timesteps,
                                  (x.size(0),), device=x.device)

                loss = self.model.compute_loss(
                    x, c_i, t, self.mean.to(x.device), self.std.to(x.device))
                total_loss += loss.item()

                self.optimizer.zero_grad(set_to_none=True)
                self.accelerator.backward(loss)
                self.optimizer.step()

            self.train_loss = total_loss / len(self.train_loader)
            self.val_loss = self.evaluate_loss(self.val_loader)
            self.log_stat()
            self.save_checkpoint(self.checkpoint_path)

            self.early_stopper(self.val_loss)
            if self.early_stopper.early_stop:
                print("Early stopping")
                break

        self.model.eval()

    def save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'mean': self.mean,
            'std': self.std,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.val_loss = checkpoint['val_loss']
        self.train_loss = checkpoint['train_loss']
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Benchmark:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def multimodality_score(self, s_l):
        ...

    def frechet_inception_distance_score(self):
        ...

    def diversity_score(self, s_d):
        ...


def euler_to_targt(x: torch.Tensor, rotation_type: RotationType) -> torch.Tensor:
    batch = x.shape[:-1]
    x = x.view(-1, 3)

    res = []

    for rotation in x:
        mat = euler_angles_to_matrix(rotation, 'XYZ')
        if rotation_type == RotationType.MATRIX:
            res.append(mat.view(-1))
        elif rotation_type == RotationType.ZHOU_6D:
            res.append(matrix_to_rotation_6d(mat))
        elif rotation_type == RotationType.QUAT:
            res.append(matrix_to_quaternion(mat))

    return torch.stack(res).reshape(*batch, -1).to(torch.float32)


@torch.no_grad()
def prepare_cmu_mocap(source_path: str, target_path: str, rotation_type: RotationType = RotationType.ZHOU_6D):
    print(f'---- Generating dataset from {source_path}')
    # returns global position and euler angles in degrees
    dataset = generate_dataset(source_path, config.block_size, 20)
    positions = dataset[:, :, :3].clone()
    rotations = dataset[:, :, 3:].clone()

    del dataset

    positions = positions - positions[:, 0].unsqueeze(1)
    rotations = euler_to_targt(rotations, rotation_type)

    mean, std = positions.mean(dim=(0, 1), keepdim=True), positions.std(
        dim=(0, 1), keepdim=True)
    positions = (positions - mean) / std
    dataset = torch.cat([positions, rotations], dim=-1)

    print(positions.shape, rotations.shape, dataset.shape)
    print("position stat", positions.min(), positions.max(),
          positions.mean(), positions.std())
    print("rotation stat", rotations.min(), rotations.max(),
          rotations.mean(), rotations.std())
    print("dataset stat", dataset.min(),
          dataset.max(), dataset.mean(), dataset.std())

    torch.save({
        'mean': mean,
        'std': std,
        'data': dataset,
        'rotation_type': rotation_type,
    }, target_path)


class DatasetSource(Enum):
    CMU = 'cmu'


def parse_arguments():
    parser = ArgumentParser()

    def dataset_source(value):
        try:
            return DatasetSource(value)
        except ValueError:
            raise ArgumentTypeError(f"Invalid dataset source. Choose from {[e.value for e in DatasetSource]}")

    parser.add_argument('--data_source', type=dataset_source, required=True,
                        help='Specify the dataset source (e.g., cmu or hm36)')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--rotation', type=RotationType,
                        default=RotationType.ZHOU_6D)

    parser.add_argument('--source_path', type=str,
                        default='data')
    parser.add_argument('--target_path', type=str,
                        default='data.pt')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.gen:
        prepare_cmu_mocap(args.source_path, args.target_path,
                          RotationType(args.rotation))

    if args.train:
        print('Training model')
        model = Diffusion()
        trainer = Trainer(model, args.source_path,
                          "checkpoint.pth", early_stopper_patience=1000, epochs=10000)
        trainer.train()
