import os
import random
import shutil
import numpy as np
import scipy
import cv2

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Could not read", path)
        return None
    return img.astype(np.float32) / (2**(img.dtype.itemsize*8) - 1)

def write_img(path, img):
    img = (img * 255).astype(np.uint8).clip(0, 255)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 98, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])

def show_img(img, name="img"):
    img = (img * 255).astype(np.uint8)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_entropy(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray_img.ravel(), bins=128, range=(0, 1))
    entropy = scipy.stats.entropy(hist / hist.sum(), base=2)
    return entropy

def pad_to_longest(img):
    # pad to the longest side to make it square
    h, w = img.shape[0:2]
    if h > w:
        pad = h - w
        img = cv2.copyMakeBorder(img, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_WRAP)
    elif w > h:
        pad = w - h
        img = cv2.copyMakeBorder(img, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_WRAP)
    return img

def process(in_dir, output_dir):
    albedo_path = None
    for path in ["albedo.jpg", "albedo.png", "base.jpg", "base.png", " diffuse.jpg", "diffuse.png",] :
        if os.path.exists(os.path.join(in_dir, path)):
            albedo_path = os.path.join(in_dir, path)
            break
    if albedo_path is None:
        print("No albedo found in", in_dir)
        return
    
    albedo = read_img(albedo_path)
    if len(albedo.shape) == 2:
        albedo = np.dstack((albedo, albedo, albedo))
    
    if not os.path.exists(os.path.join(in_dir, "normal.jpg")):
        if not os.path.exists(os.path.join(in_dir, "normal.png")):
            print("No normal.jpg or normal.png found in", in_dir)
            return
        else:
            normal_path = os.path.join(in_dir, "normal.png")
    else:
        normal_path = os.path.join(in_dir, "normal.jpg")
    normal = read_img(normal_path)

    if not os.path.exists(os.path.join(in_dir, "rough.jpg")):
        if not os.path.exists(os.path.join(in_dir, "rough.png")):
            print("No rough.jpg or rough.png found in", in_dir)
            return
        else:
            rough_path = os.path.join(in_dir, "rough.png")
    else:
        rough_path = os.path.join(in_dir, "rough.jpg")
    rough = read_img(rough_path)

    if not os.path.exists(os.path.join(in_dir, "disp.jpg")):
        if not os.path.exists(os.path.join(in_dir, "disp.png")):
            print("No disp.jpg or disp.png found in", in_dir)
            return
        else:
            disp_path = os.path.join(in_dir, "disp.png")
    else:
        disp_path = os.path.join(in_dir, "disp.jpg")
    disp = read_img(disp_path)

    if not os.path.exists(os.path.join(in_dir, "ao.jpg")):
        if not os.path.exists(os.path.join(in_dir, "ao.png")):
            ao = np.ones_like(albedo)
            return
        else:
            ao = read_img(os.path.join(in_dir, "ao.png"))
    else:
        ao = read_img(os.path.join(in_dir, "ao.jpg"))
    
    if not os.path.exists(os.path.join(in_dir, "opacity.jpg")):
        if not os.path.exists(os.path.join(in_dir, "opacity.png")):
            opacity = np.ones_like(albedo[:, :, 0:1])
        else:
            opacity = read_img(os.path.join(in_dir, "opacity.png"))
    else:
        opacity = read_img(os.path.join(in_dir, "opacity.jpg"))

    albedo = pad_to_longest(albedo)
    normal = pad_to_longest(normal)
    rough = pad_to_longest(rough)
    disp = pad_to_longest(disp)
    ao = pad_to_longest(ao)
    opacity = pad_to_longest(opacity)

    if albedo.shape[0] > 4096 and albedo.shape[1] > 4096:
        albedo = cv2.resize(albedo, (4096, 4096), interpolation=cv2.INTER_AREA)
        normal = cv2.resize(normal, (4096, 4096), interpolation=cv2.INTER_AREA)
        rough = cv2.resize(rough, (4096, 4096), interpolation=cv2.INTER_AREA)
        disp = cv2.resize(disp, (4096, 4096), interpolation=cv2.INTER_AREA)
        ao = cv2.resize(ao, (4096, 4096), interpolation=cv2.INTER_AREA)
        opacity = cv2.resize(opacity, (4096, 4096), interpolation=cv2.INTER_AREA)

    baked = albedo
    if ao is not None:
        if len(ao.shape) == 2:
            ao = np.dstack((ao, ao, ao))
        baked = baked * ao
        
    if len(rough.shape) == 2:
        rough = np.dstack((rough, rough, rough))
    if len(disp.shape) == 2:
        disp = np.dstack((disp, disp, disp))

    # many textures do not have opacity included, so we'll ignore it for now
    #albedo_op = np.dstack((albedo, opacity))
    #baked_op = np.dstack((baked, opacity))
    
    dir_name = os.path.basename(in_dir)
    if albedo.shape[0] > 2048 and albedo.shape[1] > 2048:
        save_tiled(albedo, baked, normal, rough, disp, opacity, output_dir, dir_name)

    # half resolution version
    albedo_d2 = cv2.resize(albedo, (2048, 2048), interpolation=cv2.INTER_AREA)
    baked_d2 = cv2.resize(baked, (2048, 2048), interpolation=cv2.INTER_AREA)
    normal_d2 = cv2.resize(normal, (2048, 2048), interpolation=cv2.INTER_AREA)
    rough_d2 = cv2.resize(rough, (2048, 2048), interpolation=cv2.INTER_AREA)
    disp_d2 = cv2.resize(disp, (2048, 2048), interpolation=cv2.INTER_AREA)
    opacity_d2 = cv2.resize(opacity, (2048, 2048), interpolation=cv2.INTER_AREA)

    save_tiled(albedo_d2, baked_d2, normal_d2, rough_d2, disp_d2, opacity_d2, output_dir, dir_name, 1024, suffix="_d2")

    # quarter resolution version
    albedo_d4 = cv2.resize(albedo, (1024, 1024), interpolation=cv2.INTER_AREA)
    baked_d4 = cv2.resize(baked, (1024, 1024), interpolation=cv2.INTER_AREA)
    normal_d4 = cv2.resize(normal, (1024, 1024), interpolation=cv2.INTER_AREA)
    rough_d4 = cv2.resize(rough, (1024, 1024), interpolation=cv2.INTER_AREA)
    disp_d4 = cv2.resize(disp, (1024, 1024), interpolation=cv2.INTER_AREA)
    opacity_d4 = cv2.resize(opacity, (1024, 1024), interpolation=cv2.INTER_AREA)

    save_tiled(albedo_d4, baked_d4, normal_d4, rough_d4, disp_d4, opacity_d4, output_dir, dir_name, 1024, suffix="_d4")


def save_tiled(albedo, baked, normal, rough, disp, opacity, output_dir, filename, tile_size=1024, suffix=""):
    skip_tiles = set()
    tile_size = min(tile_size, albedo.shape[0], albedo.shape[1])

    albedo_dir = os.path.join(output_dir, "albedo" + suffix)
    baked_dir = os.path.join(output_dir, "baked_ao" + suffix)
    normal_dir = os.path.join(output_dir, "normal" + suffix)
    rough_dir = os.path.join(output_dir, "rough" + suffix)
    disp_dir = os.path.join(output_dir, "disp" + suffix)

    os.makedirs(albedo_dir, exist_ok=True)
    os.makedirs(baked_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(rough_dir, exist_ok=True)
    os.makedirs(disp_dir, exist_ok=True)

    # ignore tiles more than 33% transparent
    opacity_tiles = [opacity[x:x+tile_size, y:y+tile_size] for x in range(0, opacity.shape[0],tile_size) for y in range(0, opacity.shape[1],tile_size)]
    for i, tile in enumerate(opacity_tiles):
        if np.mean(tile) < 0.66:
            skip_tiles.add(i)

    # split images into 1024x1024 tiles
    a_n = 0
    albedo_tiles = [albedo[x:x+tile_size, y:y+tile_size] for x in range(0, albedo.shape[0],tile_size) for y in range(0, albedo.shape[1],tile_size)]
    for i, tile in enumerate(albedo_tiles):
        if i in skip_tiles:
            continue
        write_img(os.path.join(albedo_dir, f"{filename}_{i}.jpg"), tile)
        a_n += 1

    b_n = 0
    baked_tiles = [baked[x:x+tile_size, y:y+tile_size] for x in range(0, baked.shape[0],tile_size) for y in range(0, baked.shape[1],tile_size)]
    for i, tile in enumerate(baked_tiles):
        if i in skip_tiles:
            continue
        write_img(os.path.join(baked_dir, f"{filename}_{i}.jpg"), tile)
        b_n += 1

    n_n = 0
    normal_tiles = [normal[x:x+tile_size, y:y+tile_size] for x in range(0, normal.shape[0],tile_size) for y in range(0, normal.shape[1],tile_size)]
    for i, tile in enumerate(normal_tiles):
        if i in skip_tiles:
            continue
        write_img(os.path.join(normal_dir, f"{filename}_{i}.jpg"), tile)
        n_n += 1

    r_n = 0
    rough_tiles = [rough[x:x+tile_size, y:y+tile_size] for x in range(0, rough.shape[0],tile_size) for y in range(0, rough.shape[1],tile_size)]
    for i, tile in enumerate(rough_tiles):
        if i in skip_tiles:
            continue
        write_img(os.path.join(rough_dir, f"{filename}_{i}.jpg"), tile)
        r_n += 1

    d_n = 0
    disp_tiles = [disp[x:x+tile_size, y:y+tile_size] for x in range(0, disp.shape[0],tile_size) for y in range(0, disp.shape[1],tile_size)]
    for i, tile in enumerate(disp_tiles):
        if i in skip_tiles:
            continue
        write_img(os.path.join(disp_dir, f"{filename}_{i}.jpg"), tile)
        d_n += 1
    
    if a_n != b_n or a_n != n_n or a_n != r_n or a_n != d_n or b_n != n_n or b_n != r_n or b_n != d_n or n_n != r_n or n_n != d_n or r_n != d_n:
        print("Mismatched tile counts: ", a_n, b_n, n_n, r_n, d_n, "for", filename)

    return

def process_dataset():
    dataset_dir = r"F:\datasets\pbr"
    output_dir = r"X:\datasets\pbr"

    for i, (root, dirs, files) in enumerate(os.walk(dataset_dir)):
        if len(files) == 0:
            continue

        process(root, output_dir)

def main():
    process_dataset()


if __name__ == '__main__':
    main()