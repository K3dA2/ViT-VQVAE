import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from einops.layers.torch import Rearrange

class Transformer(nn.Module):
    def __init__(self, emb_dim, n_heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mha1 = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 10),
            nn.GELU(),
            nn.Linear(emb_dim * 10, emb_dim)
        )       

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_output, _ = self.mha1(x_norm, x_norm, x_norm)
        x = x + attn_output

        x_norm = self.ln2(x)
        ff_output = self.ff(x_norm)
        x = x + ff_output
        return x

class EncoderViT(nn.Module):
    def __init__(self, in_channels=3, p=8, dim=128, depth=8, latent_dim = 64):
        super(EncoderViT, self).__init__()
        patch_dim = in_channels * p * p
        self.num_patches = (128 // p) * (128 // p)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.att = nn.ModuleList([Transformer(dim) for _ in range(depth)])
        self.pos_emb = nn.Embedding(self.num_patches, dim)
        self.proj = nn.Linear(dim, latent_dim)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        pos = torch.arange(0, self.num_patches, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)
        x += pos

        for layer in self.att:
            x = layer(x)

        x = self.proj(x)
        return x

class DecoderViT(nn.Module):
    def __init__(self, in_channels=3, p=8, dim=128, depth=8, latent_dim = 64):
        super(DecoderViT, self).__init__()
        patch_dim = in_channels * p * p
        self.num_patches = (128 // p) * (128 // p)
        self.to_image = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.GELU(),
            nn.Linear(patch_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = 16, w = 16,p1=p, p2=p, c=in_channels),
        )
        self.att = nn.ModuleList([Transformer(dim) for _ in range(depth)])
        self.pos_emb = nn.Embedding(self.num_patches, dim)
        self.proj = nn.Linear(latent_dim,dim)

    def forward(self, x):
        x = self.proj(x)
        pos = torch.arange(0, self.num_patches, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)
        x += pos

        for layer in self.att:
            x = layer(x)

        x = self.to_image(x)
        return x


class TestViTModel(unittest.TestCase):

    def setUp(self):
        self.encoder = EncoderViT(in_channels=3, p=8, dim=128, depth=8)
        self.decoder = DecoderViT(in_channels=3, p=8, dim=128, depth=8)

    def test_encoder_forward(self):
        batch_size = 2
        channels = 3
        height = width = 128
        input_data = torch.randn(batch_size, channels, height, width)
        output = self.encoder(input_data)
        expected_output_shape = (batch_size, 256, 64)
        self.assertEqual(output.shape, expected_output_shape)

    def test_decoder_forward(self):
        batch_size = 2
        channels = 3
        height = width = 128
        input_data = torch.randn(batch_size, 256, 64)
        output = self.decoder(input_data)
        expected_output_shape = (batch_size, height, width, channels)
        self.assertEqual(output.shape, expected_output_shape)

    def test_number_of_parameters(self):
        encoder_param_count = sum(p.numel() for p in self.encoder.parameters())
        decoder_param_count = sum(p.numel() for p in self.decoder.parameters())
        print(f"Number of parameters in the encoder: {encoder_param_count}")
        print(f"Number of parameters in the decoder: {decoder_param_count}")

if __name__ == "__main__":
    unittest.main()
