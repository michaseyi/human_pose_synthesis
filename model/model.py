import torch
from torch import nn


'''
Possible architectures
1.
    Encoder - encode bone features to latent space
    Generator - generate pose from latent space input
    Discriminator - classify real and generated poses

'''
class HumanMotionPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(

        )


        self.encoder = nn.Sequential(

        )

        self.distriminator = nn.Sequential(
            
        )
        ...

    def forward(self, x):
        ...