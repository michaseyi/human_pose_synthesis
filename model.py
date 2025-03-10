import torch
from torch import Tensor, nn
from typing import Optional, Tuple
from enum import Enum
import torch.nn.functional as F
from accelerate import Accelerator
from rotation_conversions import axis_angle_to_matrix, axis_angle_to_quaternion, matrix_to_axis_angle, quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d
from torch.utils.data import TensorDataset, DataLoader
import os
import math
from pathlib import Path


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


class FeedFowardBlock(nn.Module):
    def __init__(self, input_embedding_size: int, hidden_embedding_size: int, output_embedding_size: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_embedding_size, hidden_embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_embedding_size, output_embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EncoderBlock(nn.Module):
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

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        key_value = query
        x, _ = self.attn(self.ln_1(query), self.ln_1(
            key_value), self.ln_1(key_value))
        x += query
        x = self.mlp(self.ln_2(x)) + x
        return x


class DecoderBlock(nn.Module):
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
        self.ln_3 = nn.LayerNorm(input_embedding_size)

        self.self_attn = nn.MultiheadAttention(
            input_embedding_size, attention_head_count, dropout, batch_first=True,)

        self.cross_attn = nn.MultiheadAttention(
            input_embedding_size, attention_head_count, dropout, batch_first=True)

        self.mlp = FeedFowardBlock(
            input_embedding_size, hidden_embedding_size, output_embedding_size, dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # length = query.size(1)

        # attn_mask = torch.triu(torch.ones(length, length, device=query.device) *
        #                        float('-inf'), diagonal=1)
        attn_mask = None
        x, _ = self.self_attn(self.ln_1(query), self.ln_1(query),
                              self.ln_1(query), attn_mask=attn_mask)
        x = x + query

        x2, _ = self.cross_attn(self.ln_2(x), self.ln_2(
            key_value), self.ln_2(key_value))

        x = x2 + x

        x = self.mlp(self.ln_3(x)) + x

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_embedding_size: int,
        hidden_embedding_size: int,
        output_embedding_size: int,
        attention_head_count: int,
        dropout: float,
    ):
        super().__init__()

        # Stack of encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(
                input_embedding_size,
                hidden_embedding_size,
                output_embedding_size,
                attention_head_count,
                dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(input_embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        # Apply the final normalization
        x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_embedding_size: int,
        hidden_embedding_size: int,
        output_embedding_size: int,
        attention_head_count: int,
        dropout: float,
    ):
        super().__init__()

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(
                input_embedding_size,
                hidden_embedding_size,
                output_embedding_size,
                attention_head_count,
                dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(input_embedding_size)

        self.projection = nn.Linear(
            output_embedding_size, output_embedding_size)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        x = query

        for layer in self.layers:
            x = layer(x, key_value)

        # Apply the final normalization
        x = self.norm(x)

        # Project to vocabulary size
        logits = self.projection(x)
        return logits


class Denoiser(nn.Module):
    def __init__(
        self,
        num_joints: int,
        timesteps: int,
        rotation_type: RotationType,
        block_size: int,
        sequential: bool = True,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        attention_head_count: int = 8,
        input_embedding_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        feature_size = 3 + num_joints * rotation_type_to_dim(rotation_type)
        self.sequential = sequential

        hidden_embedding_size = 2048
        output_embedding_size = input_embedding_size

        self.encoder = Encoder(encoder_layers, input_embedding_size, hidden_embedding_size,
                               output_embedding_size, attention_head_count, dropout)
        self.decoder = Decoder(decoder_layers, input_embedding_size, hidden_embedding_size,
                               output_embedding_size, attention_head_count, dropout)
        
        self.encoder_positional_embedding = nn.Embedding(
            block_size, input_embedding_size)

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
        x_position_embd = self.positional_embedding(
            torch.arange(x.size(1), device=x.device) )

        c_feature_embd = self.feature_embedding(c)
        c_position_embd = self.encoder_positional_embedding(c_i)

        x = x_feature_embd + x_position_embd + time_embd
        c = c_feature_embd + c_position_embd + time_embd

        c = self.encoder(c)
        x = self.decoder(x, c)
        x = self.output(x)
        return x


def linear_beta_schedule(timesteps):
    # https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    # https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
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


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    # https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
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
    def __init__(
        self,
        block_size: int,
        num_joints: int,
        rotation_type: RotationType,
        skeleton_path: Path,
        reconstruction_loss_weight: float = 1.0,
        context_loss_weight: float = 1.0,
        timesteps: int = 300,
    ):
        super().__init__()

        self.block_size = block_size
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.context_loss_weight = context_loss_weight
        self.rotation_type = rotation_type
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps).to(torch.float32)
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

        self.denoiser = Denoiser(
            num_joints, timesteps, rotation_type, block_size, sequential=False)

        assert skeleton_path.exists()

        skeleton = torch.load(skeleton_path, map_location='cpu')

        self.register_buffer('parents', skeleton['parents'])
        self.register_buffer('joints', skeleton['joints'])

    def forward_diffusion_sample(self, x_0, t) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_0)
        return self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise, noise

    def extract(self, a, t, x_shape):
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1, ) * (len(x_shape) - 1)))

    def compute_loss(self, x: torch.Tensor, c_i: torch.Tensor, t: torch.Tensor):
        '''
        Loss components
            1. Global loss between predicted clean motion and ground truth
            2. Reconstruction loss of joint positions between predicted clean motion and noisy motion
            3. Reconstruction loss of joint positions between context keypoints and the same keypoints in the predicted clean motion
            4. Angular velocity loss
            5. Position velocity loss
            6. Stability loss
        '''
        c = x.gather(-2, c_i.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        x_t, _ = self.forward_diffusion_sample(x, t)
        # predicted clean motion at time t
        x_hat_t = self.denoiser(x_t, c, c_i, t)

        # Forward kinematics
        x_f = self.forward_kinematics(x)
        x_hat_f = self.forward_kinematics(x_hat_t)

        # Global loss
        l_g = F.mse_loss(x_hat_t, x).sqrt()

        # Reconstruction loss
        l_r = F.mse_loss(x_hat_f, x_f).sqrt()

        # # Context loss
        l_c = F.mse_loss(
            x_hat_f.gather(1, c_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -
                           1, *x_hat_f.shape[-2:])),
            x_f.gather(
                1, c_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *x_f.shape[-2:]))
        ).sqrt()


        # Angular Velocity loss
        x_vel = x[:, 1:] - x[:, :-1]
        x_hat_t_vel = x_hat_t[:, 1:] - x_hat_t[:, :-1]
        l_t_vel = F.mse_loss(x_hat_t_vel, x_vel).sqrt()

        # Position Velocity loss
        x_f_vel = x_f[:, 1:] - x_f[:, :-1]
        x_hat_f_vel = x_hat_f[:, 1:] - x_hat_f[:, :-1]
        l_r_vel = F.mse_loss(x_hat_f_vel, x_f_vel).sqrt()

        # Stability loss
        l_s = F.mse_loss(x_hat_f[:, 1:], x_hat_f[:, :-1]).sqrt()

        return l_g + l_r  + l_c + l_t_vel + l_r_vel

    @torch.no_grad()
    def sample(self, c: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        batch_size = c.size(0)

        motion = torch.randn(batch_size, self.block_size,
                             c.size(-1), device=c.device)

        for i in range(self.timesteps)[::-1]:
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

    def forward_kinematics(self, x):
        # x -> (b, block_size, feature_length)
        batch = x.shape[:2]
        trans = x[:, :, :3]
        poses = x[:, :, 3:]
        poses = poses.reshape(
            *batch, -1, rotation_type_to_dim(self.rotation_type))

        match self.rotation_type:
            case RotationType.ZHOU_6D:
                poses = rotation_6d_to_matrix(poses)
            case RotationType.QUAT:
                poses = quaternion_to_matrix(poses)
            case RotationType.MATRIX:
                poses = poses.reshape(*poses.shape[:-1], 3, 3)

        poses = poses.reshape(batch[0] * batch[1], *poses.shape[2:])

        positions = batch_rigid_transform(poses, self.joints.unsqueeze(
            0).repeat(poses.shape[0], 1, 1), self.parents)
        positions = positions.reshape(*batch, -1, 3)

        return positions + trans.unsqueeze(2)


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            checkpoint_path: str,
            train: TensorDataset,
            test: TensorDataset,
            val: TensorDataset,
            block_size: int,
            timesteps: int = 300,
            lr: float = 3e-4,
            epochs: int = 5000,
            batch_size: int = 64,
            early_stopper_patience=5,
    ):
        torch.manual_seed(22)

        self.epochs = epochs
        self.epoch = 0
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.early_stopper = EarlyStopping(patience=early_stopper_patience)
        self.block_size = block_size
        self.timesteps = timesteps

        optimizer = torch.optim.AdamW(model.parameters(), lr)

        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(
            val, batch_size=batch_size, shuffle=False)

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
            c_length = int(torch.randint(1, self.block_size // 2, ()).item())
            c_i = self.get_indices(self.block_size, c_length, x.size(0), x.device)
            t = torch.randint(0, self.timesteps,
                              (x.size(0),), device=x.device)
            loss = self.model.compute_loss(
                x, c_i, t)
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(data_loader)
    
    def get_indices(self, max: int, length: int, batch_size: int, device='cpu'):
        indices = []
        for _ in range(batch_size):
            if torch.rand(1).item() > 0.5: 
                idx = torch.arange(length, device=device)
            else:
                idx = torch.randperm(max, device=device)[:length]
            
            indices.append(idx)
        return torch.stack(indices)

    def train(self):
        self.model.train()
        print('---- Training')
        for _ in range(self.epoch, self.epochs):
            self.epoch += 1
            total_loss = 0

            for batch in self.train_loader:
                x = batch[0]
                c_length = int(torch.randint(
                    1, self.block_size // 2, ()).item())
              
                c_i = self.get_indices(self.block_size, c_length , x.size(0), x.device)
                t = torch.randint(0, self.timesteps,
                                  (x.size(0),), device=x.device)

                loss = self.model.compute_loss(
                    x, c_i, t)

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
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.val_loss = checkpoint['val_loss']
        self.train_loss = checkpoint['train_loss']


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


def transform_pose_datasets(path: Path, rotation: RotationType) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    train = torch.load(path.joinpath('train.pt'))
    test = torch.load(path.joinpath('test.pt'))
    val = torch.load(path.joinpath('val.pt'))

    match rotation:
        case RotationType.ZHOU_6D:
            train['poses'] = matrix_to_rotation_6d(
                axis_angle_to_matrix(train['poses']))
            val['poses'] = matrix_to_rotation_6d(
                axis_angle_to_matrix(val['poses']))
            test['poses'] = matrix_to_rotation_6d(
                axis_angle_to_matrix(test['poses']))

        case RotationType.QUAT:
            train['poses'] = axis_angle_to_quaternion(train['poses'])
            val['poses'] = axis_angle_to_quaternion(val['poses'])
            test['poses'] = axis_angle_to_quaternion(test['poses'])

        case RotationType.MATRIX:
            train['poses'] = axis_angle_to_matrix(train['poses'])
            val['poses'] = axis_angle_to_matrix(val['poses'])
            test['poses'] = axis_angle_to_matrix(test['poses'])

    return (TensorDataset(torch.cat([train['trans'], train['poses'].view(*train['poses'].shape[:2], -1)], dim=-1)),
            TensorDataset(torch.cat([test['trans'], test['poses'].view(
                *test['poses'].shape[:2], -1)], dim=-1)),
            TensorDataset(torch.cat([val['trans'], val['poses'].view(*val['poses'].shape[:2], -1)], dim=-1)))


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    # https://github.com/nghorbani/amass
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    # https://github.com/nghorbani/amass
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    return posed_joints

def get_indices(max: int, length: int, batch_size: int, device='cpu'):
        indices = []
        for _ in range(batch_size):
            if torch.rand(1).item() > 0.5: 
                idx = torch.arange(length, device=device)
            else:
                idx = torch.randperm(max, device=device)[:length]
            
            indices.append(idx)
        return torch.stack(indices)

if __name__ == "__main__":
    num_joints = 22
    rotation_type = RotationType.ZHOU_6D
    reconstruction_loss_weight = 1.5
    context_loss_weight = 1.5
    block_size = 75
    batch_size = 64
    feature_length = 135
    timesteps = 300

    train, test, val = transform_pose_datasets(
        Path('data_prepared'), rotation=rotation_type)

    model = Diffusion(block_size, num_joints, rotation_type,
                      Path('skeleton.pt'), timesteps=timesteps).to('cuda')
    
    
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])


    trainer = Trainer(model, "checkpoint.pth", train, test, val, block_size,
                      timesteps=timesteps, batch_size=batch_size, early_stopper_patience=100000)

    trainer.train()

    #inferencing
    model.eval()
    torch.seed()
    
    data = torch.load('data_prepared/DanceDB.pt')
    size = 10
    source="dancedb"
    data['poses'] = matrix_to_rotation_6d(axis_angle_to_matrix(data['poses']))
    data = torch.cat([data['trans'], data['poses'].reshape(*data['trans'].shape[:2], 22 * 6)], dim=-1).to('cuda')

    # x = train.tensors[0][index].to('cuda')
    # c_i = torch.randperm(block_size, device='cuda')[:15]
    c_i = get_indices(batch_size, size, min(200, len(data)), 'cuda')
    # start = torch.arange(30, 50, device=x.device)
    # end = torch.arange(block_size - 5, block_size, device=x.device)
    # middle = torch.arange(35, 40, device=x.device)
    # c_i = torch.cat([start], dim=-1)
    perm = torch.randperm(len(data))
    c = data[perm].gather(1, c_i.unsqueeze(-1).expand(-1, -1, 135))
    # c = data[c_i]

    out = []

    for i in range(10):
        o = model.sample(c, c_i)
        out.append(o)

    o = torch.cat(out, dim=0)

    trans = o[:, :, :3]
    poses = o[:, :, 3:]
    poses = poses.reshape(*poses.shape[:-1], -1, 6)
    poses = matrix_to_axis_angle(rotation_6d_to_matrix(poses))

    torch.save(
        {
            'indices': c_i,
            'trans': trans,
            'poses': poses,
        },
        f"prediction_{source}_{size}.pt"
    )
    print(trans.shape, poses.shape)
