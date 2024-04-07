#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GANModels.py

PyTorch Transformer GAN models

Author: Xiaomin Li, Texas State University
Date: 10/20/2022


TODOS:
* Understand the architecture and parameters of the GAN model
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor 
import math 
import numpy as np

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

#unit test
from torch.utils import data
#from MITBIH import mitbih_oneClass, mitbih_twoClass

# conv1d version
# class Generator(nn.Module):
#     '''
#     The Style Transfer Transformer generator 
#     Input: 
#         Real data signal from a majority class. Data shape = [N, ch, t]
#     Output:
#         Synthetic data signal, Dtat shape = [N, ch, t]
#     '''
#     def __init__(self, seq_len=187, channels=1, data_embed_dim=10, depth=3, num_heads=5, 
#                 forward_drop_rate=0.5, attn_drop_rate=0.5):
#         super().__init__()
#         self.seq_len = seq_len
#         self.channels = channels
#         self.data_embed_dim = data_embed_dim
#         self.depth = depth
#         self.num_heads = num_heads
#         self.attn_drop_rate = attn_drop_rate
#         self.forward_drop_rate = forward_drop_rate
        
#         self.blocks = Gen_TransformerEncoder(
#                  depth=self.depth,
#                  emb_size = self.data_embed_dim,
#                  num_heads = self.num_heads,
#                  drop_p = self.attn_drop_rate,
#                  forward_drop_p=self.forward_drop_rate
#                 )

#         self.deconv = nn.Conv1d(self.data_embed_dim, self.channels, kernel_size=1, stride=1, padding=0)
        
#         self.conv = nn.Conv1d(self.channels, self.data_embed_dim, kernel_size=1, stride=1, padding=0)


        
#     def forward(self, data): # [n, ch, t]
#         x = self.conv(data) # [n, embed_dim, t]
#         x = x.view(-1, self.seq_len, self.data_embed_dim) # [n, t, embed_dim]
#         x = self.blocks(x) # [n, t, embed_dim]
#         x = x.reshape(x.shape[0], x.shape[2], x.shape[1]) # [n, embed_dim, t]
#         output = self.deconv(x) #[n, ch, t]
#         return output

class Generator_z(nn.Module):
    def __init__(self, seq_len=187, channels=1, latent_dim=100, data_embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.data_embed_dim = data_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.data_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.data_embed_dim))
        self.blocks = Gen_TransformerEncoder(
                         depth=self.depth,
                         emb_size = self.data_embed_dim,
                         num_heads = self.num_heads,
                         drop_p = self.attn_drop_rate,
                         forward_drop_p=self.forward_drop_rate
                        )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.data_embed_dim, self.channels, 1, 1, 0)
        )

    def forward(self, z):
        x = self.l1(z).view(-1, self.seq_len, self.data_embed_dim)
        x = x + self.pos_embed # add or not add, what's the difference
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)
        return output
    
class Generator(nn.Module):
    '''
    The Style Transfer Transformer generator 
    Input: 
        Real data signal from a majority class. Data shape = [N, ch, 1, t]
    Output:
        Synthetic data signal, Dtat shape = [N, ch, 1, t]
    '''
    def __init__(self, seq_len=187, channels=1, data_embed_dim=10, depth=3, num_heads=5, 
                forward_drop_rate=0.5, attn_drop_rate=0.5, latent_dim=100):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.data_embed_dim = data_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        self.blocks = Gen_TransformerEncoder(
                 depth=self.depth,
                 emb_size = self.data_embed_dim,
                 num_heads = self.num_heads,
                 drop_p = self.attn_drop_rate,
                 forward_drop_p=self.forward_drop_rate
                )

        self.deconv = nn.Conv2d(self.data_embed_dim, self.channels, kernel_size=1, stride=1, padding=0)
        
        self.conv = nn.Conv2d(self.channels, self.data_embed_dim, kernel_size=1, stride=1, padding=0)


        
    def forward(self, data): # [n, ch, 1, t]
        x = self.conv(data) # [n, embed_dim, 1, t]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[3]) #[n, embed_dim, t]
        x = self.blocks(x.permute(0, 2, 1)) # [n, t, embed_dim]
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2]) # [n, 1, t, embed_dim]
        output = self.deconv(x.permute(0, 3, 1, 2)) #[n, ch, 1, t]
        return output
    
    
class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])       
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

        
        
class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

    
class PatchEmbedding_Linear(nn.Module):
    #what are the proper parameters set here?
    def __init__(self, in_channels = 1, patch_size = 1, emb_size = 50, seq_length = 187):
        # self.patch_size = patch_size
        super().__init__()
        #change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size),
            nn.Linear(patch_size*in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x        
        
        
class Discriminator(nn.Sequential):
    '''
    The Style Transfer Transformer discriminator 
    Input: 
        Real data or Synthetic data from generator. Data shape = [N, 1, ch, t]
    Output:
        Binary Classification, real or fake (1 or 0). Dtat shape = [N, 1]
    '''
    def __init__(self, 
                 in_channels=1,
                 patch_size=1,
                 emb_size=50, 
                 seq_length = 187,
                 depth=3, 
                 n_classes=1, 
                 **kwargs):
        print(in_channels)
        print("_______")
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )        
        
def main():
    """ Unit test code"""
#     G = Generator() 
#     G.to(torch.double)
#     twoClassdata = mitbih_twoClass(class_id1 = 0, class_id2 = 1) #data_1, labels_1, data_2, labels_2
#     train_loader = data.DataLoader(twoClassdata, batch_size=32, num_workers=4, shuffle=True)
    
#     for iter_idx, (org_sigs, org_labels, tag_sigs, tag_labels) in enumerate(train_loader):
#         output = G(org_sigs.double()) #[n, ch, 1, t]
#         print(output.shape)
#         D = Discriminator()
#         D.to(torch.double)
#         result = D(output)
#         print(result.shape)
#         if (iter_idx > 3):
#             break
    sample_size = 100
    G = Generator_z()
    z = torch.FloatTensor(np.random.normal(0, 1, (sample_size, 100)))
    print(z.shape)
    output = G(z)
    print(output.shape)
    
    D = Discriminator()
    result = D(output)
    print(result.shape)
    
if __name__ == "__main__":
    main()