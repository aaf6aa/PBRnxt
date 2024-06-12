import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.rrdbnet_arch import RRDBNet
from archs.scunetv2_arch import SCUNet

class PbrNxtNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=[3, 3, 1, 1], gen_dim=96, up_dim=32, scale=4, hr_crop=256):
        super(PbrNxtNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        self.input_size = (np.ceil(hr_crop // scale / 64) + 1) * 64
        self.gen = SCUNet(self.in_channels, self.out_channels, gen_dim, 1, 2, 2, 4, 0.1, 0.0, self.input_size, 1)

        self.ups = nn.ModuleList([nn.Identity()] * len(self.out_channels))
        for i in range(len(self.out_channels)):
            self.ups[i] = RRDBNet(in_channels + sum(self.out_channels), self.out_channels[i], up_dim, 12, upscale=scale, plus=True)

    def forward(self, x):
        pad = 32 // 2
        lr_h, lr_w = x.size(2), x.size(3)
        x = torch.roll(x, shifts=(x.size(2)//2, x.size(3)//2), dims=(2, 3))
        x = F.pad(x, (pad, pad, pad, pad), "circular")

        if self.training:
            gens, _ = self.gen(x, use_aux=False)
        else:
            gens = self.gen(x)

        srs = torch.cat([x, gens], dim=1) # temp
        srs_split = [None] * len(self.out_channels)
        for i in range(len(self.out_channels)):
            srs_split[i] = self.ups[i](srs)
            srs_split[i] = srs_split[i][..., pad*self.scale:pad*self.scale+lr_h*self.scale, pad*self.scale:pad*self.scale+lr_w*self.scale]
            srs_split[i] = torch.roll(srs_split[i], shifts=(-srs_split[i].size(2) // 2, -srs_split[i].size(3) // 2), dims=(2, 3))

        srs = torch.cat(srs_split, dim=1)

        if not self.training:
            return srs
        
        gens = gens[..., pad:pad+lr_h, pad:pad+lr_w]
        gens = torch.roll(gens, shifts=(-gens.size(2) // 2, -gens.size(3) // 2), dims=(2, 3))

        gens_split = torch.split(gens, self.out_channels, dim=1)

        return srs, srs_split, gens, gens_split
    
import time

def test_eval():
    print( "PbrNxtNet(3, [3, 3, 1, 1], 96, 32, 4, 256)")
    model = PbrNxtNet(3, [3, 3, 1, 1], 96, 32, 4, 256).to(torch.device('cuda'), dtype=torch.bfloat16, non_blocking=True)
    model.eval()
    x_cf = torch.randn(1, 3, 64, 64).to(torch.device('cuda'), dtype=torch.bfloat16, non_blocking=True)
    with torch.no_grad():
        y_cf = model(x_cf)[0]
    print(f"channels_first {x_cf.shape} -> {y_cf.shape}")
    
    if False:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
        ) as profiler:
            with torch.no_grad():
                y_cf = model(x_cf)[0]

        for s in str(profiler.key_averages()).split('\n'):
            print(s)

    batch_size = 1
    size = 512
    iters = 8

    total_time = 0
    for _ in range(iters):
        x = torch.randn(batch_size, 3, size, size).to(torch.device('cuda'), dtype=torch.bfloat16, non_blocking=True)
        start = time.time()
        with torch.no_grad():
            y = model(x)[0]
        total_time += time.time() - start
    print(f"channels_first {batch_size}x {size}x{size}: Average time: {total_time / iters * 1000:.3f}ms")
    print(f"channels_first gpu used {torch.cuda.max_memory_allocated(device=None)/1024/1024:.2f}MB memory")

    model = model.to(memory_format=torch.channels_last)
    model.eval()
    x_cl = x_cf.to(memory_format=torch.channels_last)
    with torch.no_grad():
        y_cl = model(x_cl)[0]
    print(f"channels_last {x_cl.shape} -> {y_cl.shape}")

    total_time = 0
    for _ in range(iters):
        x = torch.randn(batch_size, 3, size, size).to(torch.device('cuda'), dtype=torch.bfloat16, non_blocking=True).to(memory_format=torch.channels_last)
        start = time.time()
        with torch.no_grad():
            y = model(x)[0]
        total_time += time.time() - start
    print(f"channels_last {batch_size}x {size}x{size}: Average time: {total_time / iters * 1000:.3f}ms")
    print(f"channels_last gpu used {torch.cuda.max_memory_allocated(device=None)/1024/1024:.2f}MB memory")

    print(f"parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    test_eval()
