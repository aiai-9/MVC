#coding:utf-8

import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional

from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from mamba_ssm.modules.mamba_simple import Mamba  # Ensure correct import path for Mamba

from munch import Munch
import yaml

class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)

class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, output_padding=1, padding=1)
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
    
        return s

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for l in self.main:
            x = l(x)
            features.append(x) 
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features

class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p
        
        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == 'none':
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)
            
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    


class BiMambaTextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2), dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        # Convolutional layers for initial feature extraction
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(dropout),
            ))

        # Bi-Mamba blocks (forward and backward)
        self.mamba_f = Mamba(d_model=channels)
        self.mamba_b = Mamba(d_model=channels)

        # Projection layer to combine forward and backward outputs
        self.projection = nn.Linear(2 * channels, channels)

    def forward(self, x, input_lengths, m):
        # Embedding lookup
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]

        # Apply CNN layers
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m.unsqueeze(1), 0.0)  # Apply mask

        # Bi-Mamba processing
        x = x.transpose(1, 2)  # [B, T, chn]
        forward_output = self.mamba_f(x)
        backward_output = self.mamba_b(torch.flip(x, dims=[1]))  # Reverse for backward pass

        # Combine forward and backward outputs
        combined = torch.cat([forward_output, backward_output], dim=-1)  # [B, T, 2 * chn]
        combined = self.projection(combined)  # [B, T, chn]

        # Mask again to remove padding artifacts
        combined = combined.masked_fill(m.unsqueeze(-1), 0.0)

        return combined

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        for c in self.cnn:
            x = c(x)
        x = x.transpose(1, 2)

        # Mamba SSM inference (both forward and backward)
        forward_output = self.mamba_f(x)
        backward_output = self.mamba_b(torch.flip(x, dims=[1]))

        # Combine outputs and project
        combined = torch.cat([forward_output, backward_output], dim=-1)
        combined = self.projection(combined)

        return combined

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask





class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    
class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
                
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        
        
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)



# class ProsodyPredictor(nn.Module):

#     def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
#         super().__init__() 
        
#         self.text_encoder = DurationEncoder(sty_dim=style_dim, 
#                                             d_model=d_hid,
#                                             nlayers=nlayers, 
#                                             dropout=dropout)

#         self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
#         self.duration_proj = LinearNorm(d_hid, max_dur)
        
#         self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
#         self.F0 = nn.ModuleList([
#             AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
#             AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
#             AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
#         ])
#         self.N = nn.ModuleList([
#             AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
#             AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
#             AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
#         ])
        
#         self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
#         self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

#     def forward(self, texts, style, text_lengths, alignment, m):
#         print("Shape of texts input:", texts.shape)
#         print("Shape of style input:", style.shape)
        
#         d = self.text_encoder(texts, style, text_lengths, m)
#         print("Output shape from DurationEncoder (d):", d.shape)
        
#         batch_size = d.shape[0]
#         text_size = d.shape[1]
        
#         # Predict duration
#         input_lengths = text_lengths.cpu().numpy()
#         x = nn.utils.rnn.pack_padded_sequence(
#             d, input_lengths, batch_first=True, enforce_sorted=False)
#         print("Packed sequence shape for LSTM:", x.data.shape)
        
#         m = m.to(text_lengths.device).unsqueeze(1)
#         print("Mask shape after unsqueeze:", m.shape)
        
#         self.lstm.flatten_parameters()
#         x, _ = self.lstm(x)
#         x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
#         print("Output shape after LSTM and padding (x):", x.shape)
        
#         # Pad the output to match the mask shape
#         x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
#         x_pad[:, :x.shape[1], :] = x
#         x = x_pad
#         print("Padded output shape (x_pad):", x.shape)
        
#         # Apply dropout and project to duration
#         duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
#         print("Duration output shape after projection:", duration.shape)
        
#         # Apply alignment for duration encoding
#         print("D shape: ", d.shape)
#         print("alignment shape: ", alignment.shape)
#         en = d.transpose(-1, -2) @ alignment
#         print("Shape after alignment multiplication (en):", en.shape)

#         return duration.squeeze(-1), en
    
#     def F0Ntrain(self, x, s):
#         print("Initial shape of x in F0Ntrain:", x.shape)
        
#         # Process x through the shared LSTM
#         x, _ = self.shared(x.transpose(-1, -2))
#         print("Output shape after shared LSTM:", x.shape)
        
#         # F0 processing
#         F0 = x.transpose(-1, -2)
#         for i, block in enumerate(self.F0):
#             F0 = block(F0, s)
#             print(f"Shape after F0 block {i}:", F0.shape)
#         F0 = self.F0_proj(F0)
#         print("Final F0 shape after projection:", F0.shape)

#         # N processing
#         N = x.transpose(-1, -2)
#         for i, block in enumerate(self.N):
#             N = block(N, s)
#             print(f"Shape after N block {i}:", N.shape)
#         N = self.N_proj(N)
#         print("Final N shape after projection:", N.shape)

#         return F0.squeeze(1), N.squeeze(1)
    
#     def length_to_mask(self, lengths):
#         mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
#         mask = torch.gt(mask + 1, lengths.unsqueeze(1))
#         return mask




class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.dim_sum = d_hid + style_dim

        # Initialize DurationEncoder with Mamba blocks
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )
        
        # Add dropout to Mamba blocks to improve generalization
        self.dropout_p = 0.3  # Adjust this based on experimentation

        # Projection layer to reduce dimensions from 640 to 512
        self.to_512_proj = nn.Linear(self.dim_sum, 512)

        self.mamba_blocks = nn.ModuleList()
        for _ in range(nlayers):
            mamba_layer = nn.Sequential(
                Mamba(d_model=self.dim_sum),
                nn.Dropout(self.dropout_p)  # Add dropout here
            )
            self.mamba_blocks.append(mamba_layer)
            self.mamba_blocks.append(AdaLayerNorm(style_dim, self.dim_sum))  # Keep dimensions as 640

        self.mamba_ssm = Mamba(d_model=self.dim_sum)

        # Projection layers for duration, F0, and N
        self.duration_proj = LinearNorm(512, max_dur)  # Update to match 512 dimensions
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

        # Define F0 and N branches with AdaIN blocks
        self.F0 = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        ])
        self.N = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        ])

        # Linear projection layer to ensure consistent output shape
        self.projection = nn.Linear(self.dim_sum, d_hid)
        self.dropout = dropout

    def forward(self, texts, style, text_lengths, alignment, m):
        # Pass through DurationEncoder
        d = self.text_encoder(texts, style, text_lengths, m)
        print(f"\nShape after DurationEncoder: {d.shape}")

        # Sequentially apply mamba_blocks
        for i, block in enumerate(self.mamba_blocks):
            if isinstance(block, Mamba):
                d = block(d)
            elif isinstance(block, AdaLayerNorm):
                d = block(d, style)  # Apply AdaLayerNorm without projection
            print(f"After block {i + 1}: d shape = {d.shape}")

            # Apply dropout and mask
            d = F.dropout(d, p=self.dropout, training=self.training)
            batch_size, seq_len, feature_dim = d.shape
            m_expanded = m.unsqueeze(-1).expand(-1, -1, feature_dim)
            d = d.masked_fill(m_expanded, 0.0)
            print(f"After dropout and masking in block {i + 1}: d shape = {d.shape}")

        # # Project d to 512 dimensions before passing to duration_proj
        # d = self.to_512_proj(d)
        # print(f"Shape after projection to 512 dimensions: {d.shape}")

        # Apply alignment for duration encoding
        print("D shape before duration: ", d.shape)

        # Prepare mask and padding for duration projection
        m = m.to(text_lengths.device).unsqueeze(-1)
        x_pad = torch.zeros(batch_size, d.shape[1], d.shape[2], device=d.device)
        x_pad[:, :d.shape[1], :] = d
        m_expanded = m.expand(batch_size, d.shape[1], d.shape[2])

        # Apply mask
        x = x_pad.masked_fill(m_expanded == 1, 0.0)
        x = self.to_512_proj(x)
        print("padded output shape (x):", x.shape)

        # Project for duration prediction
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        # duration = self.duration_proj(F.dropout(d, p=0.5, training=self.training))
        print(f"Shape after duration projection: {duration.shape}")

        # Apply alignment for duration encoding
        print("D shape: ", d.shape)
        print("alignment shape: ", alignment.shape)
        en = d.transpose(-1, -2) @ alignment
        print("Shape after alignment multiplication (en):", en.shape)

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        print("Initial shape of x in F0Ntrain:", x.shape)

        # Process x through the Mamba layer and project down to 512 dimensions
        x = self.mamba_ssm(x.transpose(-1, -2))
        print("Shape after Mamba layer in F0Ntrain:", x.shape)

        # Project Mamba output to expected dimension for AdainResBlk1d
        x = self.projection(x).transpose(-1, -2)  # Shape: [batch_size, 512, seq_len]
        print("Shape after projection in F0Ntrain:", x.shape)

        # F0 processing
        F0 = x
        for i, block in enumerate(self.F0):
            F0 = block(F0, s)
            print(f"Shape after F0 block {i}:", F0.shape)
        F0 = self.F0_proj(F0)
        print("Final F0 shape after projection:", F0.shape)

        # N processing
        N = x
        for i, block in enumerate(self.N):
            N = block(N, s)
            print(f"Shape after N block {i}:", N.shape)
        N = self.N_proj(N)
        print("Final N shape after projection:", N.shape)

        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask





# class ProsodyPredictor(nn.Module):
#     def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
#         super().__init__()

#         # Initialize DurationEncoder
#         self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)

#         # Mamba-based shared modeling
#         self.mamba_shared = Mamba(d_model=d_hid + style_dim)

#         # Project to 512 dimensions instead of 640
#         self.to_512_proj = nn.Linear(d_hid + style_dim, 512)  # Project output to 512 dimensions
#         self.duration_proj = LinearNorm(512, max_dur)

#         # Adjust the F0 and N branches with AdainResBlk1d to use 512 dimensions
#         self.F0 = nn.ModuleList([
#             AdainResBlk1d(512, 512, style_dim, dropout_p=dropout),
#             AdainResBlk1d(512, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
#             AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
#         ])
#         self.N = nn.ModuleList([
#             AdainResBlk1d(512, 512, style_dim, dropout_p=dropout),
#             AdainResBlk1d(512, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
#             AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
#         ])

#         # Projection layers for F0 and N
#         self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
#         self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

#     def forward(self, texts, style, text_lengths, alignment, m):
#         print("\n\nShape of texts input:", texts.shape)
#         print("Shape of style input:", style.shape)

#         # Process through DurationEncoder
#         d = self.text_encoder(texts, style, text_lengths, m)
#         print("Output shape from DurationEncoder (d):", d.shape)

#         # Ensure d has 3D shape before Mamba
#         batch_size, text_size, _ = d.shape
#         d = self.mamba_shared(d)
#         print("Output shape after shared Mamba (d):", d.shape)

#         # Project to 512 dimensions
#         d = self.to_512_proj(d.view(batch_size, text_size, -1))
#         print("Projected shape to 512 dimensions:", d.shape)

#         # Prepare mask and padding for duration projection
#         m = m.to(text_lengths.device).unsqueeze(-1)
#         x_pad = torch.zeros(batch_size, d.shape[1], d.shape[2], device=d.device)
#         x_pad[:, :d.shape[1], :] = d
#         m_expanded = m.expand(batch_size, d.shape[1], d.shape[2])

#         # Apply mask
#         x = x_pad.masked_fill(m_expanded == 1, 0.0)
#         print("Masked and padded output shape (x):", x.shape)

#         # Duration projection
#         duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
#         print("Duration output shape after projection:", duration.shape)

#         # Apply alignment for duration encoding
#         en = d.transpose(-1, -2) @ alignment
#         print("Shape after alignment multiplication (en):", en.shape)

#         return duration.squeeze(-1), en

#     def F0Ntrain(self, x, s):
#         print("Initial shape of x in F0Ntrain:", x.shape)

#         # Process x through Mamba shared layer and projection to 512 dimensions
#         x = self.to_512_proj(self.mamba_shared(x.transpose(-1, -2))).transpose(-1, -2)
#         print("Shape after Mamba and projection in F0Ntrain:", x.shape)

#         # F0 processing
#         F0 = x
#         for i, block in enumerate(self.F0):
#             F0 = block(F0, s)
#             print(f"Shape after F0 block {i}:", F0.shape)
#         F0 = self.F0_proj(F0)
#         print("Final F0 shape after projection:", F0.shape)

#         # N processing
#         N = x
#         for i, block in enumerate(self.N):
#             N = block(N, s)
#             print(f"Shape after N block {i}:", N.shape)
#         N = self.N_proj(N)
#         print("Final N shape after projection:", N.shape)

#         return F0.squeeze(1), N.squeeze(1)




#     def length_to_mask(self, lengths):
#         mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
#         mask = torch.gt(mask + 1, lengths.unsqueeze(1))
#         return mask






    
class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.mamba_blocks = nn.ModuleList()
        
        for i in range(nlayers):
            # print(f"Initializing Mamba block {i+1} and AdaLayerNorm {i+1}")
            # Replace LSTM with Mamba, initialized with the same dimensions
            self.mamba_blocks.append(Mamba(d_model=d_model + sty_dim))
            self.mamba_blocks.append(AdaLayerNorm(sty_dim, d_model))

        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim
        # print("DurationEncoder with Mamba initialization complete.")
        # print(f"d_model: {d_model}, sty_dim: {sty_dim}, nlayers: {nlayers}, dropout: {dropout}")

    def forward(self, x, style, text_lengths, m):
        # print("Entering forward pass of Mamba-integrated DurationEncoder.")
        # print("Initial x shape:", x.shape)
        # print("Initial style shape:", style.shape)

        masks = m.to(text_lengths.device)
        # print("Mask shape after transfer to device:", masks.shape)

        # Reshape and expand style for concatenation
        x = x.permute(2, 0, 1)  # [T, B, d_model]
        # print("x shape after permute:", x.shape)

        s = style.expand(x.shape[0], x.shape[1], -1)  # Match temporal dimension
        # print("Expanded style shape:", s.shape)

        # Concatenate style to x and apply mask
        x = torch.cat([x, s], axis=-1)  # Concatenate on feature dimension
        # print("x shape after concatenation with style:", x.shape)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        # print("x shape after masked fill:", x.shape)

        x = x.transpose(0, 1)  # [B, T, d_model + sty_dim]
        # print("x shape after transpose for Mamba input:", x.shape)

        x = x.transpose(-1, -2)  # Prepare shape for Mamba
        
        # Process through each Mamba and AdaLayerNorm block
        for i, block in enumerate(self.mamba_blocks):
            if isinstance(block, AdaLayerNorm):
                # print(f"Applying AdaLayerNorm {i//2 + 1}")

                # Split x into feature and style parts
                x_feature = x[:, :self.d_model, :]  # Get the feature part
                x_style = x[:, self.d_model:, :]    # Get the style part

                # Apply AdaLayerNorm to the feature part only
                x_feature = block(x_feature.transpose(-1, -2), style).transpose(-1, -2)
                # print(f"x_feature shape after AdaLayerNorm {i//2 + 1}:", x_feature.shape)

                # Concatenate the normalized feature part with the style part
                x = torch.cat([x_feature, x_style], axis=1)
                # print(f"x shape after re-concatenation with style part {i//2 + 1}:", x.shape)

                # Apply mask after re-concatenation
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
                # print(f"x shape after masked fill in AdaLayerNorm {i//2 + 1}:", x.shape)
            else:
                # print(f"Applying Mamba block {i//2 + 1}")

                # Pass through Mamba
                x = block(x.transpose(-1, -2))
                x = x.transpose(-1, -2)  # Transpose back after Mamba
                # print(f"x shape after Mamba block {i//2 + 1}:", x.shape)

                # Apply dropout
                x = F.dropout(x, p=self.dropout, training=self.training)
                # print(f"x shape after dropout {i//2 + 1}:", x.shape)

                # Prepare padded tensor to match expected dimensions
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad
                # print(f"x shape after padding {i//2 + 1}:", x.shape)


                # print("Exiting forward pass of Mamba-integrated DurationEncoder.")
                return x.transpose(-1, -2)  # Final transpose to match expected output shape

    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        # print("Mask shape:", mask.shape)
        return mask

    
def load_F0_models(path):
    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    
    return F0_model

def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model

def build_model(args, text_aligner, pitch_extractor, bert):
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
    if args.decoder.type == "istftnet":
        from Modules.istftnet import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
    else:
        from Modules.hifigan import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
        
    text_encoder = BiMambaTextEncoder(
                                        channels=args.hidden_dim,
                                        kernel_size=5,
                                        depth=args.n_layer,
                                        n_symbols=args.n_token
                                    )

    
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
        
    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    context_features=args.style_dim*2, 
                                    **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    **args.diffusion.transformer)
    
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
        channels=args.style_dim*2,
        context_features=args.style_dim*2,
    )
    
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0 
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    
    nets = Munch(
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            predictor=predictor,
            decoder=decoder,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            diffusion=diffusion,

            text_aligner = text_aligner,
            pitch_extractor=pitch_extractor,

            mpd = MultiPeriodDiscriminator(),
            msd = MultiResSpecDiscriminator(),
        
            # slm discriminator head
            wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
       )
    
    return nets

def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key], strict=False)
    _ = [model[key].eval() for key in model]
    
    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0
        
    return model, optimizer, epoch, iters

