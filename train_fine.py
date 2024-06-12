import math
import time
import albumentations as A
import cv2
import numpy as np
import random
import textile
import torch
import torch.utils.data

from archs.discriminator_arch import MultiscaleUNetDiscriminator
from data import ADXTCompression, ARandomOrder, MultiFolderDataset, tensor_to_img
from losses import DISTSLoss, WeightedLoss, GanLoss
import brdf
from pbrnxt_net import PbrNxtNet

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def loss_dict_to_str(loss_dict):
    return ", ".join([f"{k}: {v:.2e}" for k, v in loss_dict.items()])

def pad(x, pad=8):
    _, _, h, w = x.size()
    pad_left = math.ceil(pad / 2)
    pad_right = math.floor(pad / 2)
    pad_top = math.ceil(pad / 2)
    pad_bottom = math.floor(pad / 2)
    x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), "reflect")

    return x, (pad_top, pad_left, h, w)

def unpad(x, pads, scale=4):
    pad_top, pad_left, h, w = pads
    x = x[..., pad_top*scale:pad_top*scale+h*scale, pad_left*scale:pad_left*scale+w*scale]

    return x

def train(
        network,
        network_d,
        dataloader,
        validation_dataloader=None,
        n_iters=10000,
        lr=1e-3,
        lr_d=1e-4,
        min_lr=1e-6,
        lr_steps=2000,
        lr_steps_mult=2,
        virtual_batch_size=1,
        pix_weight=0.01,
        perceptual_weight=0.0,
        aux_weight=0.1,
        deg_weight=0.1,
        tile_weight=0.0,
        render_weight=0.0,
        gan_weight=0.0,
        out_channels=[3, 3, 1, 1],
        scale=4,
        hr_crop=256,
        device=torch.device('cuda'),
        network_name="network"
        ):
    network = network.train()

    loss_fn = torch.nn.HuberLoss()

    if render_weight > 0:
        renderer = brdf.Renderer(device=device)

    loss_main = WeightedLoss(loss_fn, pix_weight)
    if tile_weight > 0:
        loss_textile = textile.Textile(resolution=(hr_crop, hr_crop), number_tiles=3)

    if perceptual_weight > 0:
        loss_percep = WeightedLoss(DISTSLoss(True).bfloat16().to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last), perceptual_weight)

    if network_d is not None:
        loss_gan = WeightedLoss(GanLoss(), gan_weight)
        loss_gan_d = WeightedLoss(GanLoss(), 1.0)

    optimizer = torch.optim.NAdam(network.parameters(), lr, betas=[0.98, 0.99], weight_decay=0.01, decoupled_weight_decay=True)
    if lr_steps is None or lr_steps == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters, min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, lr_steps, lr_steps_mult, min_lr)
    scaler = torch.cuda.amp.GradScaler(init_scale=2.**5)
    
    if network_d is not None:
        network_d = network_d.train()
        optimizer_d = torch.optim.NAdam(network_d.parameters(), lr_d, betas=[0.98, 0.99], weight_decay=0.01, decoupled_weight_decay=True)
        if lr_steps is None or lr_steps == 0:
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, n_iters, min_lr)
        else:
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_d, lr_steps, lr_steps_mult, min_lr)
        scaler_d = torch.cuda.amp.GradScaler(init_scale=2.**5)

    epoch = 0
    total_iter = 0
    running_loss = 0.0
    running_loss_list = []
    time_start = time.time()
    cv_shown = False

    try:
        while (total_iter < n_iters):
            for _, data in enumerate(dataloader):
                img_lrs, img_hrs = data
                img_lr = img_lrs[0].to(torch.device('cuda'), non_blocking=True).bfloat16().to(memory_format=torch.channels_last)
                img_hrs_split = [img.to(torch.device('cuda'), non_blocking=True).bfloat16().to(memory_format=torch.channels_last) for img in img_hrs]

                img_hrs = torch.cat(img_hrs_split, dim=1)
                img_hrs_down = torch.nn.functional.interpolate(img_hrs, scale_factor=1/scale, mode='bilinear', align_corners=False)

                loss_dict = {}
                loss_total = torch.tensor(0.0).cuda()

                loss_dict["pix"] = 0
                loss_dict["tile"] = 0
                if deg_weight > 0:
                    loss_dict["deg"] = 0

                if render_weight > 0:
                    loss_dict["r_pix"] = 0

                if perceptual_weight > 0:
                    loss_dict["percep"] = 0
                    if render_weight > 0:
                        loss_dict["r_percep"] = 0
                if network_d is not None:
                    loss_dict["gan"] = 0
                    loss_dict["gan_d"] = 0

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    img_srs, img_srs_split, img_gens, img_gens_split = network(img_lr)

                    if network_d is not None:
                        for p in network_d.parameters():
                            p.requires_grad = False
                        loss_gan_val = loss_gan(network_d(img_srs), True)
                        loss_gan_val += loss_gan(network_d(img_gens), True) * aux_weight
                        loss_dict["gan"] += loss_gan_val
                        loss_total += loss_gan_val

                    loss_pix_val = loss_main(img_srs, img_hrs)
                    loss_pix_val += loss_main(img_gens, img_hrs_down) * aux_weight
                    loss_dict["pix"] += loss_pix_val
                    loss_total += loss_pix_val

                    if deg_weight > 0:
                        img_srs_down = torch.nn.functional.interpolate(img_srs, scale_factor=1/scale, mode='bilinear', align_corners=False)
                        loss_deg_val = loss_main(img_srs_down, img_hrs_down) * deg_weight
                        loss_deg_val += loss_main(img_srs_down, img_gens) * deg_weight * aux_weight
                        loss_dict["deg"] += loss_deg_val
                        loss_total += loss_deg_val

                    if render_weight > 0:
                        renderer.step()
                        render_sr = renderer.render(img_srs_split[0], img_srs_split[1], img_srs_split[2], img_srs_split[3])
                        render_hr = renderer.render(img_hrs_split[0], img_hrs_split[1], img_hrs_split[2], img_hrs_split[3])
                        
                        loss_render_val = loss_main(render_sr, render_hr) * render_weight
                        loss_dict["r_pix"] += loss_render_val
                        loss_total += loss_render_val
                        
                        if perceptual_weight > 0:
                            loss_render_percep_val = loss_percep(render_sr, render_hr) * render_weight
                            loss_dict["r_percep"] += loss_render_percep_val
                            loss_total += loss_render_percep_val
                        
                    for i, (img_gen, img_sr, img_hr) in enumerate(zip(img_gens_split, img_srs_split, img_hrs_split)):
                        if img_sr.size(1) == 1:
                            img_sr = img_sr.repeat(1, 3, 1, 1)
                            img_gen = img_gen.repeat(1, 3, 1, 1)
                        if tile_weight > 0:
                            loss_tile_val = (1.0 - loss_textile(img_sr).mean().item()) * tile_weight
                            loss_tile_val += (1.0 - loss_textile(img_gen).mean().item()) * tile_weight * aux_weight
                            loss_dict["tile"] += loss_tile_val

                        if perceptual_weight > 0:
                            loss_percep_val = loss_percep(img_sr, img_hr.detach())
                            loss_percep_val += loss_percep(img_gen, torch.nn.functional.interpolate(img_hr, scale_factor=1/scale, mode='bilinear', align_corners=False)) * aux_weight
                            loss_dict["percep"] += loss_percep_val

                    loss_dict["tile"] /= len(img_srs_split) # mean
                    loss_total += loss_dict["tile"]
                    if perceptual_weight > 0:
                        loss_dict["percep"] /= len(img_srs_split) # mean
                        loss_total += loss_dict["percep"]
                       
                scaler.scale(loss_total).backward()
                
                if total_iter % virtual_batch_size == 0 or total_iter >= n_iters:
                    scaler.step(optimizer)
                    optimizer.step()
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                if network_d is not None:
                    # discriminator training
                    for p in network_d.parameters():
                        p.requires_grad = True

                    loss_total_d = 0
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        # real
                        loss_total_d += loss_gan_d(network_d(img_hrs), True)
                        # fake
                        loss_total_d += loss_gan_d(network_d(img_srs.detach().clone()), False)
                        loss_total_d += loss_gan_d(network_d(img_gens.detach().clone()), False) * aux_weight

                    loss_dict["gan_d"] += loss_total_d
                    scaler_d.scale(loss_total_d).backward()
                            
                    if total_iter % virtual_batch_size == 0 or total_iter >= n_iters:
                        scaler_d.step(optimizer_d)
                        optimizer_d.step()
                        scaler_d.update()
                        scheduler_d.step()
                        optimizer_d.zero_grad()

                # print statistics
                total_iter += 1

                running_loss_list.append(loss_total.item())
                while len(running_loss_list) > 100:
                    running_loss_list.pop(0)
                running_loss = sum(running_loss_list) / len(running_loss_list)

                if total_iter % 100 == 0:
                    time_elapsed = time.time() - time_start
                    time_start = time.time()
                    print(f'[epoch {epoch}, iter {total_iter}], Loss: {running_loss:.2e}, elapsed: {time_elapsed:.2f}s, lr: {scheduler.get_last_lr()[0]:.2e}')
                    print(loss_dict_to_str(loss_dict))

                    if True:
                        if img_srs_split is None:
                            img_srs_split = torch.split(img_srs, out_channels, dim=1)

                        cv2.imshow('albedo_lr', tensor_to_img(img_lr[0]))

                        cv2.imshow('albedo_sr', tensor_to_img(img_srs_split[0][0]))
                        cv2.imshow('normal_sr', tensor_to_img(img_srs_split[1][0]))
                        cv2.imshow('rough_sr', tensor_to_img(img_srs_split[2][0]))
                        cv2.imshow('disp_sr', tensor_to_img(img_srs_split[3][0]))
                        cv2.imshow('albedo_hr', tensor_to_img(img_hrs_split[0][0]))
                        cv2.imshow('normal_hr', tensor_to_img(img_hrs_split[1][0]))
                        cv2.imshow('rough_hr', tensor_to_img(img_hrs_split[2][0]))
                        cv2.imshow('disp_hr', tensor_to_img(img_hrs_split[3][0]))
                        if render_weight > 0:
                            cv2.imshow('render_sr', tensor_to_img(render_sr[0]))
                            cv2.imshow('render_hr', tensor_to_img(render_hr[0]))
                        cv2.waitKey(1)
                        cv_shown = True

                if total_iter % 1000 == 0 and validation_dataloader is not None:
                    val_psnr = validate(network, validation_dataloader)
                    print(f'[epoch {epoch}, iter {total_iter}], Val PSNR: {val_psnr}')
                if total_iter % 5000 == 0:
                    torch.save(network.state_dict(), f"checkpoints/{network_name}_{total_iter}.pth")
                    if network_d is not None:
                        torch.save(network_d.state_dict(), f"checkpoints/{network_name}_d_{total_iter}.pth")
                    print(f"[epoch {epoch}, iter {total_iter:5d}], Loss: {running_loss:.2e}, saved {network_name}_{total_iter}.pth")
                if total_iter >= n_iters:
                    break
            # epoch end
            print(f"[epoch {epoch}, iter {total_iter:5d}], Loss: {running_loss:.2e}")
            epoch += 1

    except KeyboardInterrupt:
        torch.save(network.state_dict(), f"checkpoints/{network_name}_{total_iter}.pth")
        if network_d is not None:
            torch.save(network_d.state_dict(), f"checkpoints/{network_name}_d_{total_iter}.pth")
        print(f"[epoch {epoch}, iter {total_iter:5d}], Loss: {running_loss:.2e}, saved {network_name}_{total_iter}.pth")
        print(loss_dict_to_str(loss_dict))

    if cv_shown:
        cv2.destroyAllWindows()

    return network, network_d

def psnr_fn(gt, pred):
    mse = torch.mean((gt - pred) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def validate(network, dataloader):
    network = network.eval()

    total_imgs = 0
    running_psnr = 0
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            img_lrs, img_hrs = data
            img_lr = img_lrs[0].cuda().float()
            img_hrs = torch.cat([img.cuda().float() for img in img_hrs], dim=1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img_srs = network(img_lr)
                
                running_psnr += psnr_fn(img_hrs, img_srs).item()
                total_imgs += 1

    network = network.train()

    running_psnr = running_psnr / total_imgs
    return running_psnr

def test(network, dataloader, out_channels=[3,3,1,1], network_name="network"):
    network = network.eval()

    total_imgs = 0
    running_psnr = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img_lrs, img_hrs = data
            img_lr = img_lrs[0].cuda().float()
            img_hrs_split = [img.cuda().float() for img in img_hrs]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img_srs = network(img_lr)
                img_srs_split = torch.split(img_srs, out_channels, dim=1)

            batch_size = img_lr.size(0)
            for j, (img_sr, img_hr) in enumerate(zip(img_srs_split, img_hrs_split)):
                for b in range(batch_size):
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_lr.png', tensor_to_img(img_lr[b]))
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_sr{j}.png', tensor_to_img(img_sr[b]))
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_hr{j}.png', tensor_to_img(img_hr[b]))
            
                running_psnr += psnr_fn(img_hr, img_sr).item()
                total_imgs += 1

    network = network.train()

    running_psnr = running_psnr / total_imgs
    print(f'Test PSNR: {running_psnr:.2f}')

    return running_psnr

def main():
    network_name = "pbrnxt"
    out_channels = [3, 3, 1, 1]
    scale = 4
    hr_crop = 256 - (32 * scale)

    shared_transform = A.Compose([
        A.ChannelShuffle(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=True, p=1.0),
    ])
    lr_transform = A.Compose([
        ARandomOrder([
            #A.UnsharpMask(p=0.1),
            #A.GaussianBlur(p=0.1),
            A.GaussNoise(var_limit=(0,50), p=0.1),
        ]),
        ARandomOrder([
            ADXTCompression(p=1.0),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
        ]),
    ])
    
    hr_paths = [
        [r"X:\datasets\pbr\baked_ao"] + [r"X:\datasets\pbr\baked_ao_d2"] * 4 + [r"X:\datasets\pbr\baked_ao_d4"] * 16,
        [r"X:\datasets\pbr\normal"]   + [r"X:\datasets\pbr\normal_d2"] * 4   + [r"X:\datasets\pbr\normal_d4"] * 16,
        [r"X:\datasets\pbr\rough"]    + [r"X:\datasets\pbr\rough_d2"] * 4    + [r"X:\datasets\pbr\rough_d4"] * 16,
        [r"X:\datasets\pbr\disp"]     + [r"X:\datasets\pbr\disp_d2"] * 4     + [r"X:\datasets\pbr\disp_d4"] * 16,
    ]

    train_dataset = MultiFolderDataset(
        hr_paths, enlarge_ratio=1, cutmix_n=0, normal_index=1, channels=[3, 3, 1, 1],
        shared_scale=(hr_crop/1024, 1.0), lr_scale=1/scale, hr_crop=hr_crop,
        shared_transform=shared_transform, lr_transform=lr_transform)

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - 128, 64, 64])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=1)

    network = PbrNxtNet(in_channels=3, out_channels=out_channels, gen_dim=96, up_dim=32, scale=scale, hr_crop=hr_crop).to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last)
    network.load_state_dict(torch.load(f"checkpoints/pbrnxt_12483.pth"), strict=False)
 
    network_d = None
    network_d = MultiscaleUNetDiscriminator(sum(out_channels), 64, 2).to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last)
    network_d.load_state_dict(torch.load(f"checkpoints/pbrnxt_d_12483.pth"), strict=False)

    network, network_d = train(network, network_d, train_dataloader, val_dataloader, 
                               lr=1e-5, lr_d=1e-5, n_iters=200000, lr_steps=8000, lr_steps_mult=2, virtual_batch_size=1,
                               pix_weight=0.01, perceptual_weight=1.0,
                               render_weight=0.0, gan_weight=0.005, aux_weight=0.1, deg_weight=0.1,
                               tile_weight=0.1,
                               out_channels=out_channels, device=torch.device('cuda'),
                               hr_crop=hr_crop, scale=scale, network_name=network_name)

    test(network, test_dataloader, out_channels, network_name)

    return


if __name__ == "__main__":
    main()
