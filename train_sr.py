import math
import time
import albumentations as A
import cv2
import numpy as np
import random
import torch
import torch.utils.data

from archs.discriminator_arch import MultiscaleUNetDiscriminator
from archs.rrdbnet_arch import RRDBNet

from data import ADXTCompression, ARandomOrder, MultiFolderDataset, tensor_to_img
from losses import DISTSLoss, WeightedLoss, GanLoss

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.benchmark = True

def loss_dict_to_str(loss_dict):
    return ", ".join([f"{k}: {v:.2e}" for k, v in loss_dict.items()])

def padded_forward(model, x, pad=8, scale=4):
    _, _, h, w = x.size()
    pad_left = math.ceil(pad / 2)
    pad_right = math.floor(pad / 2)
    pad_top = math.ceil(pad / 2)
    pad_bottom = math.floor(pad / 2)
    x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), "reflect")

    x = model(x)
    x = x[..., pad_top*scale:pad_top*scale+h*scale, pad_left*scale:pad_left*scale+w*scale]

    return x

def train(
        networks,
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
        gan_weight=0.0,
        out_channels=[3, 3, 1, 1],
        scale=4,
        device=torch.device('cuda'),
        network_name="network"
        ):
    for network in networks:
        network.train()

    loss_main = WeightedLoss(torch.nn.HuberLoss(), pix_weight)

    if perceptual_weight > 0:
        loss_percep = WeightedLoss(DISTSLoss(True).bfloat16().to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last), perceptual_weight)

    if network_d is not None:
        loss_gan = WeightedLoss(GanLoss(), gan_weight)
        loss_gan_d = WeightedLoss(GanLoss(), 1.0)

    params = [p for network in networks for p in network.parameters()]
    optimizer = torch.optim.NAdam(params, lr, betas=[0.98, 0.99], weight_decay=0.01, decoupled_weight_decay=True)
    if lr_steps is None or lr_steps == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters, min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, lr_steps, lr_steps_mult, min_lr)
    scaler = torch.cuda.amp.GradScaler(init_scale=2.**5)
    
    if network_d is not None:
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
                img_lrs_split = [img.to(torch.device('cuda'), non_blocking=True).bfloat16().to(memory_format=torch.channels_last) for img in img_lrs]
                img_hrs_split = [img.to(torch.device('cuda'), non_blocking=True).bfloat16().to(memory_format=torch.channels_last) for img in img_hrs]

                img_lrs = torch.cat(img_lrs_split[0:1] + img_lrs_split, dim=1)
                img_hrs = torch.cat(img_hrs_split, dim=1)

                loss_dict = {}
                loss_total = torch.tensor(0.0).cuda()

                loss_dict["pix"] = 0

                if perceptual_weight > 0:
                    loss_dict["percep"] = 0
                if network_d is not None:
                    loss_dict["gan"] = 0
                    loss_dict["gan_d"] = 0

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    img_srs_split = [padded_forward(network, img_lrs, pad=8, scale=scale) for network in networks]
                    img_srs = torch.cat(img_srs_split, dim=1)

                    if network_d is not None:
                        for p in network_d.parameters():
                            p.requires_grad = False
                        pred_fake = network_d(img_srs)
                        loss_gan_val = loss_gan(pred_fake, True)
                        loss_dict["gan"] += loss_gan_val
                        loss_total += loss_gan_val

                    loss_pix_val = loss_main(img_srs, img_hrs)
                    loss_dict["pix"] += loss_pix_val
                    loss_total += loss_pix_val
                        
                    if perceptual_weight > 0:
                        for i, (img_sr, img_hr) in enumerate(zip(img_srs_split, img_hrs_split)):
                            loss_percep_val = loss_percep(img_sr, img_hr.detach())
                            loss_dict["percep"] += loss_percep_val

                        loss_dict["percep"] /= len(img_srs_split)
                        loss_total += loss_dict["percep"]
                       
                scaler.scale(loss_total).backward()
                
                if total_iter % virtual_batch_size == 0 or total_iter >= n_iters:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                if network_d is not None:
                    # discriminator training
                    for p in network_d.parameters():
                        p.requires_grad = True

                    loss_total_d = 0
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        pred_real = network_d(img_hrs)
                        loss_total_d += loss_gan_d(pred_real, True)

                        pred_fake = network_d(img_srs.detach().clone())
                        loss_total_d += loss_gan_d(pred_fake, False)

                    loss_dict["gan_d"] += loss_total_d
                    scaler_d.scale(loss_total_d).backward()
                            
                    if total_iter % virtual_batch_size == 0 or total_iter >= n_iters:
                        scaler_d.step(optimizer_d)
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

                        cv2.imshow('albedo_lr', tensor_to_img(img_lrs_split[0][0]))
                        cv2.imshow('normal_lr', tensor_to_img(img_lrs_split[1][0]))
                        cv2.imshow('rough_lr', tensor_to_img(img_lrs_split[2][0]))
                        cv2.imshow('disp_lr', tensor_to_img(img_lrs_split[3][0]))

                        cv2.imshow('albedo_sr', tensor_to_img(img_srs_split[0][0]))
                        cv2.imshow('normal_sr', tensor_to_img(img_srs_split[1][0]))
                        cv2.imshow('rough_sr', tensor_to_img(img_srs_split[2][0]))
                        cv2.imshow('disp_sr', tensor_to_img(img_srs_split[3][0]))
                        cv2.imshow('albedo_hr', tensor_to_img(img_hrs_split[0][0]))
                        cv2.imshow('normal_hr', tensor_to_img(img_hrs_split[1][0]))
                        cv2.imshow('rough_hr', tensor_to_img(img_hrs_split[2][0]))
                        cv2.imshow('disp_hr', tensor_to_img(img_hrs_split[3][0]))
                        cv2.waitKey(1)
                        cv_shown = True

                if total_iter % 1000 == 0 and validation_dataloader is not None:
                    val_psnr = validate(networks, validation_dataloader, scale=scale)
                    print(f'[epoch {epoch}, iter {total_iter}], Val PSNR: {val_psnr}')
                if total_iter % 5000 == 0:
                    for i, network in enumerate(networks):
                        torch.save(network.state_dict(), f"checkpoints/{network_name}_{i}_{total_iter}.pth")
                    if network_d is not None:
                        torch.save(network_d.state_dict(), f"checkpoints/{network_name}_d_{total_iter}.pth")
                    print(f"[epoch {epoch}, iter {total_iter:5d}], Loss: {running_loss:.2e}, saved {network_name}_{total_iter}.pth")
                if total_iter >= n_iters:
                    break
            # epoch end
            print(f"[epoch {epoch}, iter {total_iter:5d}], Loss: {running_loss:.2e}")
            epoch += 1

    except KeyboardInterrupt:
        for i, network in enumerate(networks):
            torch.save(network.state_dict(), f"checkpoints/{network_name}_{i}_{total_iter}.pth")
        if network_d is not None:
            torch.save(network_d.state_dict(), f"checkpoints/{network_name}_d_{total_iter}.pth")
        print(f"[epoch {epoch}, iter {total_iter:5d}], Loss: {running_loss:.2e}, saved {network_name}_{total_iter}.pth")
        print(loss_dict_to_str(loss_dict))

    if cv_shown:
        cv2.destroyAllWindows()

    return networks, network_d

def psnr_fn(gt, pred):
    mse = torch.mean((gt - pred) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def validate(networks, dataloader, scale=4):
    for network in networks:
        network.eval()

    total_imgs = 0
    running_psnr = 0
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            img_lrs, img_hrs = data
            img_lrs = torch.cat([img.cuda().float() for img in img_lrs], dim=1)
            img_hrs = torch.cat([img.cuda().float() for img in img_hrs], dim=1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img_srs_split = [padded_forward(network, img_lrs, pad=8, scale=scale) for network in networks]
                img_srs = torch.cat(img_srs_split, dim=1)
                
                running_psnr += psnr_fn(img_hrs, img_srs).item()
                total_imgs += 1

    for network in networks:
        network.train()
    running_psnr = running_psnr / total_imgs
    return running_psnr

def test(
        networks,
        dataloader,
        network_name="network",
        scale=4
        ):
    for network in networks:
        network.eval()

    total_imgs = 0
    running_psnr = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img_lrs, img_hrs = data
            img_lrs_split = [img.cuda().float() for img in img_lrs]
            img_hrs_split = [img.cuda().float() for img in img_hrs]
            img_lrs = torch.cat(img_lrs_split, dim=1)
            img_hrs = torch.cat(img_hrs_split, dim=1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img_srs_split = [padded_forward(network, img_lrs, pad=16, scale=scale) for network in networks]
                img_srs = torch.cat(img_srs_split, dim=1)

            batch_size = img_lrs.size(0)
            for j, (img_lr, img_sr, img_hr) in enumerate(zip(img_lrs_split, img_srs_split, img_hrs_split)):
                for b in range(batch_size):
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_lr{j}.png', tensor_to_img(img_lr[b]))
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_sr{j}.png', tensor_to_img(img_sr[b]))
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_hr{j}.png', tensor_to_img(img_hr[b]))
            
            running_psnr += psnr_fn(img_hrs, img_srs).item()
            total_imgs += 1

    running_psnr = running_psnr / total_imgs
    print(f'Test PSNR: {running_psnr:.2f}')

    return running_psnr

def main():
    network_name = "pbrnxt_up"
    in_channels = 3
    out_channels = [3, 3, 1, 1]
    scale = 4
    hr_crop = 192

    shared_transform = A.Compose([
        A.ChannelShuffle(p=0.5),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.5, always_apply=True, p=1.0),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
    ])
    lr_transform = A.Compose([
        ARandomOrder([
            A.Sharpen(p=0.1),
            A.GaussianBlur(blur_limit=(1,3), p=0.1),
            A.GaussNoise(var_limit=(0,50), p=0.5),
        ]),
        ARandomOrder([
            ADXTCompression(p=1.0),
            A.Posterize(num_bits=(6, 7), p=0.2),
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

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=1)

    networks = [None] * len(out_channels)
    for i in range(len(out_channels)):
        if i == 0:
            networks[i] = RRDBNet(in_channels + sum(out_channels), out_channels[i], 64, 23, scale, True).to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last)
        else:
            networks[i] = RRDBNet(in_channels + sum(out_channels), out_channels[i], 32, 12, scale, True).to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last)
        #networks[i] = RRDBNet(sum(out_channels), out_channels[i], 32, 12, scale, True).to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last)
        #networks[i].load_state_dict(torch.load(f"checkpoints/pbrnxt_up_{i}_157850.pth"), strict=False)

    network_d = None
    network_d = MultiscaleUNetDiscriminator(sum(out_channels), 64, 3).to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last)
    #network_d.load_state_dict(torch.load(f"checkpoints/pbrnxt_up_d_157850.pth"), strict=False)

    networks, network_d = train(networks, network_d, train_dataloader, val_dataloader, 
                               lr=1e-4, lr_d=1e-4, n_iters=200000, lr_steps=4000, lr_steps_mult=2, virtual_batch_size=1,
                               pix_weight=0.01, perceptual_weight=1.0, gan_weight=0.01,
                               out_channels=out_channels, device=torch.device('cuda'),
                               scale=scale, network_name=network_name)

    test(networks, test_dataloader, network_name, scale)

    return


if __name__ == "__main__":
    main()
