from functools import partial
import torch
import torch.nn as nn

from vision_transformer import PatchEmbed, Block, CBlock, PatchEmbed_F
from util.pos_embed import get_2d_sincos_pos_embed
from util.basic_var import SelfAttention, FFN

import torchvision.utils as vutils
import os


class MaskGenerator(nn.Module):
    def __init__(self, channels):
        super(MaskGenerator, self).__init__()
        self.freq_cut = nn.Parameter(torch.tensor(0.5)) 
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(2.0))
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.channels = channels
        self.conv[0].weight.data.fill_(0.01)

    def forward(self, x):
        base_mask = self.conv(x).squeeze(0)
        
        c, h, w = x.shape[-3:]
        center_x, center_y = w // 2, h // 2
        xx, yy = torch.meshgrid(
            torch.arange(w, device=x.device), 
            torch.arange(h, device=x.device)
        )
        dist_sq = (xx - center_x)**2 + (yy - center_y)** 2
        
        max_radius = torch.sqrt(torch.tensor(w**2 + h**2, device=x.device)) / 2

        r1 = self.freq_cut * max_radius
        
        constrained_alpha = torch.sigmoid(self.alpha) 
        constrained_beta = torch.exp(self.beta) 
        
        radial_decay = constrained_alpha * torch.exp(
            -constrained_beta * (dist_sq / (r1**2 + 1e-8))
        )

        radial_decay = radial_decay.unsqueeze(0) 
        radial_decay = radial_decay.repeat(c, 1, 1) 
        radial_decay = radial_decay.permute(0, 2, 1) 
        
        final_mask = base_mask * radial_decay
        
        return final_mask

class MaskedAutoencoderInfMAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 var_embed_dim=768, var_num_heads=12, var_mlp_ratio=4.):
        super().__init__()
        # --------------------------------------------------------------------------
        # Encoder
        self.patch_embed = PatchEmbed_F(img_size[0], patch_size[0] * patch_size[1] * patch_size[2], in_chans, embed_dim[2])

        self.patch_embed1 = PatchEmbed(img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock(dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth[2])])
        self.norm = norm_layer(embed_dim[-1])

        # --------------------------------------------------------------------------
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size[0] * patch_size[1] * patch_size[2])**2 * in_chans, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------
        # DDG
        self.var_transformers = nn.ModuleList([
            SelfAttention(block_idx=i, embed_dim=var_embed_dim, num_heads=var_num_heads) 
            for i in range(3)
        ])
        self.var_ffns = nn.ModuleList([
            FFN(in_features=var_embed_dim, hidden_features=var_embed_dim * var_mlp_ratio) 
            for _ in range(3)
        ])

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m.bias, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        p = 16
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


    # entropy
    def random_masking(self, x, mask_ratio):
        """
        Mask the top-ranked tokens based on entropy values.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches

        # Calculate entropy for each token
        probs = torch.softmax(x, dim=-1)  # [N, L, D]
        epsilon = 1e-9
        probs = torch.clamp(probs, min=epsilon)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # [N, L]

        # Sort tokens by entropy (descending)
        ids_shuffle = torch.argsort(entropy, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Calculate number of tokens to mask
        num_mask = int(L * mask_ratio)

        # Get indices of tokens to keep (low entropy ones)
        ids_keep = ids_shuffle[:, num_mask:]  # Keep tokens with lower entropy

        # Step 5: Create binary mask: 0 = keep, 1 = mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, num_mask:] = 0

        # Step 6: Unshuffle mask to original token order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def FastFourierTransformLayer(self, x):
        if x.dim() == 3:  # Assuming the shape [C, H, W]
            x = x.unsqueeze(0)  # Add a batch dimension
        
        original_dtype = x.dtype  # Save original dtype to restore later if needed

        # Convert x to float32 for FFT compatibility
        x = x.to(torch.float32)

        # Perform FFT
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft)
    
        channels = x_fft.shape[1] 
        mask_generator = MaskGenerator(channels).to(device='cuda')
        mask = mask_generator(x_fft.abs())
    
        # Apply filter to remove low-frequency parts
        x_fft_filtered = x_fft * mask
    
        # Perform iFFT
        x_ifft = torch.fft.ifftshift(x_fft_filtered)
        x_ifft = torch.fft.ifft2(x_ifft)
        x_ifft = x_ifft.real  # Take the real part

        # Convert back to original dtype if necessary
        if original_dtype != torch.float32:
            x_ifft = x_ifft.to(original_dtype)
        
        if x_ifft.shape[0] == 1:  # Check if the batch dimension is still 1
            x_ifft = x_ifft.squeeze(0)  # Remove the batch dimension

        return x_ifft


    def forward_encoder(self, x, mask_ratio):
        x_ = self.patch_embed(x)

        x_f = self.FastFourierTransformLayer(x_)

        ids_keep, mask, ids_restore = self.random_masking(x_, mask_ratio)
        mask_for_patch1 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)
        mask_for_patch2 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1)

        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)

        x = x + self.pos_embed
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))

        for blk in self.blocks3:
            x = blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, x_f, mask, ids_restore

    def forward_decoder(self, x, x_f, ids_restore):
        # Step 1: VAR Multi-Scale Feature Generation
        downsample_conv = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=4, stride=4, padding=0)
        downsample_conv = downsample_conv.to(device='cuda').half()
        x_f = x_f.permute(0, 2, 1) 
        x_f = downsample_conv(x_f)
        x_f = x_f.permute(0, 2, 1)
        var_features = []
        for i, transformer in enumerate(self.var_transformers):
            if i == 0:
                weight_x = nn.Parameter(torch.tensor(0.5))
                weight_x_f = nn.Parameter(torch.tensor(0.5))
                combined_feature = weight_x * x + weight_x_f * x_f
                # combined_feature = torch.cat([x, x_f], dim=-1)  # Concatenate along the feature dimension
                # print("x shape:", combined_feature.shape)
                var_feature = transformer(combined_feature, attn_bias=None)
            else:
                var_feature = transformer(x, attn_bias=None)
            var_feature = self.var_ffns[i](var_feature)
            var_features.append(var_feature)

        x = torch.cat(var_features, dim=1)

        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, x_f, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, x_f, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def infmae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderInfMAE(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
infmae_vit_base_patch16 = infmae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
