import cv2
import gc
import imageio
import os
import time
import torch
import torch.utils.data
from tqdm import tqdm

from data import img_to_tensor, tensor_to_img, tiled_forward
from pbrnxt_net import PbrNxtNet

if not torch.cuda.is_available():
    print('CUDA is not available. Exiting...')
    exit()

default_device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')

if torch.cuda.is_bf16_supported():
    default_dtype = torch.bfloat16
else:
    props = torch.cuda.get_device_properties(default_device)
    # fp16 supported at compute 5.3 and above
    if props.major > 5 or (props.major == 5 and props.minor >= 3):
        default_dtype = torch.float16
    else:
        default_dtype = torch.float32

def test(
        network,
        in_folder, out_folder,
        network_name="network",
        out_channels=[3,3,1,1],
        scale=4,
        names=["diffuse", "normal", "roughness", "displacement"],
        ):
    network.eval()

    files = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f))]

    total_time = 0
    total_images = 0
    with torch.no_grad():
        tq = tqdm(files)
        for file in tq:
            tq.set_description(f"Processing {file}")
            bgr2rgb = False
            try:
                if file.endswith(".dds"):
                    img_lr = imageio.v2.imread(os.path.join(in_folder, file), format='DDS')
                else:
                    bgr2rgb = True
                    img_lr = cv2.imread(os.path.join(in_folder, file), cv2.IMREAD_UNCHANGED)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            if img_lr.shape[0] <= 32 or img_lr.shape[1] <= 32:
                continue

            alpha = None
            if img_lr.ndim == 3 and img_lr.shape[2] == 4:
                alpha = img_lr[:, :, 3]
            
            img_lr = img_to_tensor(img_lr, bgr2rgb=bgr2rgb, channels=3)

            img_lr = img_lr.unsqueeze(0).to(default_device, default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
            start = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(default_dtype is not torch.float32), dtype=default_dtype):
                img_srs, _ = tiled_forward(network, img_lr, overlap=32, scale=scale, max_tile_size=1024)
                img_srs_split = torch.split(img_srs, out_channels, dim=1)
            total_time += time.time() - start
            total_images += 1

            basename, ext = os.path.splitext(file)
            os.makedirs(os.path.join(out_folder, basename), exist_ok=True)

            for name, img_sr in zip(names, img_srs_split):
                if name == "diffuse":
                    img_sr = tensor_to_img(img_sr[0])
                    if alpha is not None:
                        img_sr = cv2.cvtColor(img_sr, cv2.COLOR_BGR2BGRA)
                        img_sr[:, :, 3] = cv2.resize(alpha, (img_sr.shape[1], img_sr.shape[0]), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(out_folder, basename, f"{basename}_{network_name}_{name}.png"), img_sr)
                elif name == "normal":
                    # normalize normal map
                    normal = (img_sr[0] * 2.0 - 1.0)
                    normal = normal / torch.norm(normal, dim=0, keepdim=True)
                    normal = (normal + 1.0) / 2.0

                    cv2.imwrite(os.path.join(out_folder, basename, f"{basename}_{network_name}_{name}_gl.png"), tensor_to_img(normal))
                    # flip green channel to convert from OpenGL to DirectX normal map
                    normal[1] = 1.0 - normal[1]
                    cv2.imwrite(os.path.join(out_folder, basename, f"{basename}_{network_name}_{name}_dx.png"), tensor_to_img(normal))
                elif name == "displacement":
                    cv2.imwrite(os.path.join(out_folder, basename, f"{basename}_{network_name}_displacement.png"), tensor_to_img(img_sr[0]))
                else:
                    cv2.imwrite(os.path.join(out_folder, basename, f"{basename}_{network_name}_{name}.png"), tensor_to_img(img_sr[0]))

            torch.cuda.empty_cache()
            gc.collect()

    print(f"{network_name} {total_images} images: Average time: {total_time / total_images * 1000:.3f}ms")

    return

def main():
    
    scale = 4
    network_name = "pbrnxt"
    out_channels = [3, 3, 1, 1]
    network = PbrNxtNet(3, out_channels, 96, 32, scale, 512)
    network = network.to(default_device, default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
    network.eval()

    network.load_state_dict(torch.load(f"checkpoints/pbrnxt_294474.pth"), strict=False)

    in_folders = [
        r"X:\datasets\pbr\val\lr",
    ]
    out_folders = [
        r"X:\datasets\pbr\val\sr_pbrnxt_294474",
    ]

    for in_folder, out_folder in zip(in_folders, out_folders):
        test(network, in_folder, out_folder, network_name, out_channels, scale)

    return


if __name__ == "__main__":
    main()
