import time
import albumentations as A
import cv2
import numpy as np
import random
import torch
import torch.utils.data

from archs.discriminator_arch import MultiscaleUNetDiscriminator
from archs.scunetv2_arch import SCUNet

from data import ADXTCompression, ARandomOrder, MultiFolderDataset, tensor_to_img
from losses import DISTSLoss, WeightedLoss, GanLoss
import brdf

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.benchmark = True

def loss_dict_to_str(loss_dict):
    return ", ".join([f"{k}: {v:.2e}" for k, v in loss_dict.items()])

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
        render_weight=0.0,
        aux_weight=0.0,
        gan_weight=0.0,
        map_weights=[0.5, 1.0, 1.0, 1.0],
        out_channels=[3, 3, 1, 1],
        hr_crop=256,
        device=torch.device('cuda'),
        network_name="network"
        ):
    network.train()

    loss_fn = torch.nn.HuberLoss()

    if render_weight > 0:
        renderer = brdf.Renderer(device=device)

    loss_main = WeightedLoss(loss_fn, pix_weight)

    if perceptual_weight > 0:
        """
        layer_weights = {
            'conv1_2': 0.1,
            'conv2_2': 0.1,
            'conv3_4': 1,
            'conv4_4': 1,
            'conv5_4': 1,
        }
        loss_percep = WeightedLoss(PerceptualLoss(layer_weights, 'vgg19_bn', criterion=loss_fn).bfloat16().to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last), perceptual_weight)
        """
        loss_percep = WeightedLoss(DISTSLoss(True).bfloat16().to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last), perceptual_weight)

    if aux_weight > 0:
        loss_aux = WeightedLoss(loss_fn, aux_weight)

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
        optimizer_d = torch.optim.NAdam(network_d.parameters(), lr_d, betas=[0.98, 0.99], weight_decay=0.01, decoupled_weight_decay=True)
        if lr_steps is None or lr_steps == 0:
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters, min_lr)
        else:
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, lr_steps, lr_steps_mult, min_lr)
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

                loss_dict = {}
                loss_total = torch.tensor(0.0).cuda()

                if render_weight > 0:
                    loss_dict["r_pix"] = 0

                loss_dict["pix"] = 0

                if perceptual_weight > 0:
                    loss_dict["percep"] = 0
                    if render_weight > 0:
                        loss_dict["r_percep"] = 0
                if network_d is not None:
                    loss_dict["gan"] = 0
                    loss_dict["gan_d"] = 0
                if aux_weight > 0:
                    loss_dict["aux"] = 0
                    if render_weight > 0:
                        loss_dict["r_aux"] = 0

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    img_srs, img_auxs = network(img_lr, use_aux=aux_weight > 0)

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
                    
                    if not isinstance(img_auxs, list):
                        img_auxs = [img_auxs]
                        
                    if aux_weight > 0:
                        loss_aux_val = 0
                        for aux in img_auxs:
                            loss_aux_val += loss_aux(aux, img_hrs)
                        loss_aux_val /= len(img_auxs)
                        loss_dict["aux"] += loss_aux_val
                        loss_total += loss_aux_val
                        
                    # split the outputs into the separate images
                    if render_weight > 0 or perceptual_weight > 0:
                        img_srs_split = torch.split(img_srs, out_channels, dim=1)
                    else:
                        img_srs_split = None
                    
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
                        if aux_weight > 0:
                            loss_render_aux_val = 0
                            for aux in img_auxs:
                                aux_split = torch.split(aux, out_channels, dim=1)

                                renderer.step()
                                render_aux = renderer.render(aux_split[0], aux_split[1], aux_split[2], aux_split[3])
                                render_hr_aux = renderer.render(
                                    torch.nn.functional.interpolate(img_hrs_split[0], (aux.size(-2), aux.size(-1)), mode='bilinear', align_corners=False),
                                    torch.nn.functional.interpolate(img_hrs_split[1], (aux.size(-2), aux.size(-1)), mode='bilinear', align_corners=False),
                                    torch.nn.functional.interpolate(img_hrs_split[2], (aux.size(-2), aux.size(-1)), mode='bilinear', align_corners=False),
                                    torch.nn.functional.interpolate(img_hrs_split[3], (aux.size(-2), aux.size(-1)), mode='bilinear', align_corners=False)
                                )
                                loss_render_aux_val += loss_aux(render_aux, render_hr_aux) * render_weight
                            loss_render_aux_val /= len(img_auxs)
                            loss_dict["r_aux"] += loss_render_aux_val
                            loss_total += loss_render_aux_val

                    if perceptual_weight > 0:
                        for i, (img_sr, img_hr) in enumerate(zip(img_srs_split, img_hrs_split)):
                            loss_percep_val = loss_percep(img_sr, img_hr.detach())
                            loss_dict["percep"] += loss_percep_val * map_weights[i]

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

                        cv2.imshow('img_lr', tensor_to_img(img_lr[0]))
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
    network.eval()

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

    network.train()
    running_psnr = running_psnr / total_imgs
    return running_psnr

def test(
        network,
        dataloader,
        network_name="network",
        out_channels=[3,3,1,1],
        ):
    network.eval()

    renderer = brdf.Renderer(device=torch.device('cuda'))

    total_imgs = 0
    running_psnr = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img_lrs, img_hrs = data
            img_lr = img_lrs[0].cuda().float()
            img_hrs_split = [img.cuda().float() for img in img_hrs]
            img_hrs = torch.cat(img_hrs_split, dim=1)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img_srs = network(img_lr)
                img_srs_split = torch.split(img_srs, out_channels, dim=1)

                renderer.step()
                render_sr = renderer.render(img_srs_split[0], img_srs_split[1], img_srs_split[2], img_srs_split[3])
                render_hr = renderer.render(img_hrs_split[0], img_hrs_split[1], img_hrs_split[2], img_hrs_split[3])

            batch_size = img_lr.size(0)
            for j, (img_sr, img_hr) in enumerate(zip(img_srs_split, img_hrs_split)):
                for b in range(batch_size):
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}___lr.png', tensor_to_img(img_lr[b]))
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_sr{j}.png', tensor_to_img(img_sr[b]))
                    cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_hr{j}.png', tensor_to_img(img_hr[b]))
            for b in range(batch_size):
                cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_render_sr.png', tensor_to_img(render_sr[b]))
                cv2.imwrite(f'test/img_{i * batch_size + b}_{network_name}_render_hr.png', tensor_to_img(render_hr[b]))

            running_psnr += psnr_fn(img_hrs, img_srs).item()
            total_imgs += 1

    running_psnr = running_psnr / total_imgs
    print(f'Test PSNR: {running_psnr:.2f}')

    return running_psnr

def main():
    network_name = "pbrnxt"
    out_channels = [3, 3, 1, 1]
    scale = 1
    hr_crop = 128

    shared_transform = A.Compose([
        A.ChannelShuffle(p=0.5),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.5, always_apply=True, p=1.0),
    ])
    lr_transform = A.Compose([
        ARandomOrder([
            A.Sharpen(p=0.1),
            A.GaussianBlur(blur_limit=(1,3), p=0.05),
            A.GaussNoise(var_limit=(0,50), p=0.5),
        ]),
        ARandomOrder([
            ADXTCompression(p=1.0),
            A.Posterize(num_bits=(6, 7), p=0.2),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
        ])
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

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=1)

    input_size = (np.ceil(hr_crop // scale / 64) + 1) * 64
    network = SCUNet(3, out_channels, 96, 1, 2, 2, 4, 0.1, 0.0, input_size, scale).to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last)
    network.load_state_dict(torch.load(f"checkpoints/pbrnxt_39244.pth"), strict=False)

    network_d = None
    network_d = MultiscaleUNetDiscriminator(sum(out_channels), 64, 3).to(torch.device('cuda'), non_blocking=True).to(memory_format=torch.channels_last)
    network_d.load_state_dict(torch.load(f"checkpoints/pbrnxt_d_39244.pth"), strict=False)

    network, network_d = train(network, network_d, train_dataloader, val_dataloader, 
                               lr=1e-5, lr_d=1e-5, n_iters=200000, lr_steps=4000, lr_steps_mult=2, virtual_batch_size=1,
                               pix_weight=0.01, perceptual_weight=1.0,
                               render_weight=0.0, aux_weight=0.001,
                               gan_weight=0.01, map_weights=[1.0, 1.0, 1.0, 1.0],
                               out_channels=out_channels, hr_crop=hr_crop, device=torch.device('cuda'),
                               network_name=network_name)

    test(network, test_dataloader, network_name)

    return


if __name__ == "__main__":
    main()
