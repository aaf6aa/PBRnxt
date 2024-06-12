from typing import Callable
import albumentations as A
import cv2
import imageio
import numpy as np
import os
import pylnk3
import random
import torch.utils.data
import typing

class ADXTCompression(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(ADXTCompression, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        encoded = imageio.imwrite(imageio.RETURN_BYTES, img, format='DDS', compression='S3TC_DXT1')
        img = imageio.imread(encoded, format='DDS')
        return img

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()

class ARandomOrder(A.BaseCompose):
    def __init__(self, transforms: typing.Sequence[typing.Union[A.BasicTransform, "A.BaseCompose"]], p: float = 0.5):
        super().__init__(transforms, p)

    def __call__(self, *args, **data) -> typing.Dict[str, typing.Any]:
        random.shuffle(self.transforms)
        for t in self.transforms:
            data = t(**data)
        return data


IMG_EXTENSIONS = [".bmp", ".jpeg", ".jpg", ".jp2", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".tiff", ".tif", ".exr", ".hdr", ".pic"]

def retrieve_imgs(path):
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path

    imgs = []
    for path in paths:
        if os.path.splitext(path)[1].lower() == ".lnk":
            lnk = pylnk3.parse(path)
            path = lnk.path

        if os.path.isdir(path):
            imgs += retrieve_imgs([os.path.join(path, x) for x in os.listdir(path)])
        elif os.path.splitext(path)[1].lower() in IMG_EXTENSIONS:
            imgs.append(path)

    return imgs

def fix_normal_rotate(normal: np.ndarray, rotatecode: int) -> np.ndarray:    
    b, g, r = cv2.split(normal)
    if rotatecode == cv2.ROTATE_90_CLOCKWISE:
        normal = cv2.merge((b, r, 1 - g))
    elif rotatecode == cv2.ROTATE_180:
        normal = cv2.merge((b, 1 - g, 1 - r))
    elif rotatecode == cv2.ROTATE_90_COUNTERCLOCKWISE:
        normal = cv2.merge((b, 1 - r, g))
    return normal

def fix_normal_flip(normal: np.ndarray, flipcode: int) -> np.ndarray:
    b, g, r = cv2.split(normal)
    if flipcode == 0:
        normal = cv2.merge((b, g, 1 - r))
    elif flipcode > 0:
        normal = cv2.merge((b, 1 - g, r))
    elif flipcode < 0:
        normal = cv2.merge((b, 1 - g, 1 - r))
    return normal

def img_to_tensor(img: np.ndarray, bgr2rgb: bool = True, channels: int = 3, range: float = 1.0) -> torch.Tensor:
    max_value = np.iinfo(img.dtype).max

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if channels == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channels == 1:
        img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=-1)
    img = img * (range / max_value)
    
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))
    return img

def tensor_to_img(tensor: torch.Tensor, rgb2bgr: bool = True, dtype: np.dtype = np.uint8, normalize: bool = False) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor[0]
    max_value = np.iinfo(dtype).max

    if normalize:
        t_min, t_max = tensor.min(), tensor.max()
        tensor = (tensor - t_min)/(t_max - t_min)
    
    img = tensor.float().detach().clamp_(0, 1).permute(1, 2, 0)
    img = img * max_value
    img = img.cpu().numpy().astype(dtype)
    if rgb2bgr:
        color = cv2.COLOR_RGBA2BGRA if img.shape[2] == 4 else cv2.COLOR_RGB2BGR
        img = cv2.cvtColor(img, color)
    return img

def apply_shared_transform(imgs, transform: A.Compose = None):
    if transform is None:
        return imgs
    
    transformed_imgs = []
    for img in imgs:
        python_rand_state = random.getstate()
        torch_rand_state = torch.random.get_rng_state()
        np_rand_state = np.random.get_state()

        try:
            transformed = transform(image=img)
        except Exception as e:
            print(e)
            transformed = {'image': img}
        transformed_imgs.append(transformed['image'])

        random.setstate(python_rand_state)
        torch.random.set_rng_state(torch_rand_state)
        np.random.set_state(np_rand_state)

    return transformed_imgs

class SingleFolderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hr_paths=None,
            enlarge_ratio=1,
            channels=1, shared_scale=(0.15, 1.0), lr_scale=[0.25, 0.5, 1.0], lr_deg_count=1, hr_crop=256,
            shared_transform: A.Compose = None, hr_transform: A.Compose = None, lr_transform: A.Compose = None,
            batch_size=1
            ):
        self.hr_paths = retrieve_imgs(hr_paths)

        self.enlarge_ratio = enlarge_ratio
        self.channels = channels
        self.shared_scale = shared_scale
        self.lr_scale = lr_scale
        self.lr_deg_count = lr_deg_count
        self.hr_crop = hr_crop
        self.shared_transform = shared_transform
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.batch_size = batch_size

        self.val = False

    def get_item(self, index, scale=1.0):
        if self.val: 
            # deterministic validation transformations
            python_rand_state = random.getstate()
            torch_rand_state = torch.random.get_rng_state()
            np_rand_state = np.random.get_state()
            random.seed(index * int(scale + 1))
            torch.manual_seed(index * int(scale + 1))
            np.random.seed(index * int(scale + 1))

        index = index % len(self.lr_paths)
        img_hrs = [cv2.imread(paths[index], cv2.IMREAD_UNCHANGED) for paths in self.hr_paths]
        
        last = None
        for img_hr in img_hrs:
            if last is not None:
                if img_hr.shape[0] != last.shape[0] or img_hr.shape[1] != last.shape[1]:
                    raise ValueError("Images must have the same shape")
            last = img_hr

        img_lr_ = cv2.imread(self.lr_paths[index], cv2.IMREAD_UNCHANGED)

        shared_scale = random.uniform(self.shared_scale[0], self.shared_scale[1])
        if img_hr.shape[1] * shared_scale < self.hr_crop or img_hr.shape[0] * shared_scale < self.hr_crop:
            shared_scale = max(self.hr_crop / img_hr.shape[1], self.hr_crop / img_hr.shape[0])

        img_hrs = [cv2.resize(img_hr, (int(img_hr.shape[1] * shared_scale), int(img_hr.shape[0] * shared_scale)), interpolation=cv2.INTER_AREA) for img_hr in img_hrs]
        img_lr_ = cv2.resize(img_lr_, (int(img_lr_.shape[1] * shared_scale), int(img_lr_.shape[0] * shared_scale)), interpolation=cv2.INTER_AREA)

        # random 90 degree rotations and flip
        rotate = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
        flip = random.choice([None, 0, 1, -1])

        if rotate is not None and flip is not None:
            img_hrs_ = []
            for img_hr in img_hrs:
                if rotate is not None:
                    img_hr = cv2.rotate(img_hr, rotate)
                if flip is not None:
                    img_hr = cv2.flip(img_hr, flip)
                img_hrs_.append(img_hr)
                    
            if rotate is not None:
                img_lr_ = cv2.rotate(img_lr_, rotate)
            if flip is not None:
                img_lr_ = cv2.flip(img_lr_, flip)
            
            img_hrs = img_hrs_
            

        if self.shared_transform is not None:
            imgs = apply_shared_transform([img_lr_, img_hrs[0]], self.shared_transform)
            img_lr_ = imgs[0]
            img_hrs[0] = imgs[1]

        hr_crop_x = random.randint(0, img_hrs[0].shape[1] - self.hr_crop)
        hr_crop_y = random.randint(0, img_hrs[0].shape[0] - self.hr_crop)

        img_hrs = [img_hr[hr_crop_y:hr_crop_y+self.hr_crop, hr_crop_x:hr_crop_x+self.hr_crop] for img_hr in img_hrs]
        img_lr_ = img_lr_[hr_crop_y:hr_crop_y+self.hr_crop, hr_crop_x:hr_crop_x+self.hr_crop]

        interp_modes = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        img_lrs = [cv2.resize(img_lr_, (int(img_lr_.shape[1] * scale), int(img_lr_.shape[0] * scale)), interpolation=random.choice(interp_modes)) for _ in range(self.lr_deg_count)]

        if self.hr_transform is not None:
            img_hrs = [self.hr_transform(image=img_hr)['image'] for img_hr in img_hrs]

        if self.lr_transform is not None:
            img_lrs = [self.lr_transform(image=img_lr)['image'] for img_lr in img_lrs]

        if self.val:
            random.setstate(python_rand_state)
            torch.random.set_rng_state(torch_rand_state)
            np.random.set_state(np_rand_state)
        
        img_lrs = [img_to_tensor(img_lr, channels=self.channels[0]) for img_lr in img_lrs]
        img_hrs = [img_to_tensor(img_hr, channels=num_channels) for img_hr, num_channels in zip(img_hrs, self.channels)]

        return img_lrs, img_hrs 
    
    def get_batch(self, index, batch_size=1):
        t = type(self.lr_scale)
        if t is list:
            lr_scale = random.choice(self.lr_scale)
        elif t is tuple:
            lr_scale = random.uniform(self.lr_scale[0], self.lr_scale[1])
        else:
            lr_scale = self.lr_scale

        batch_lrs = []
        batch_hrs = []
        for _ in range(batch_size):
            img_lrs, img_hrs = self.get_item(index, lr_scale)
            batch_lrs.append(img_lrs)
            batch_hrs.append(img_hrs)
            index = random.randint(0, len(self.lr_paths) - 1)
        
        batch_lr = [torch.stack([img_lr[i] for img_lr in batch_lrs]) for i in range(self.lr_deg_count)]
        batch_hrs = [torch.stack([img_hr[i] for img_hr in batch_hrs]) for i in range(len(self.hr_paths))]

        return batch_lr, batch_hrs, 1 / lr_scale

    def __getitem__(self, index):
        return self.get_batch(index, self.batch_size)

    def __len__(self):
        return len(self.lr_paths) * self.enlarge_ratio

class MultiFolderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hr_paths=[None, None],
            enlarge_ratio=1, cutmix_n=3, normal_index=1,
            channels=1, shared_scale=(0.15, 1.0), lr_scale=0.25, hr_crop=256,
            shared_transform: A.Compose = None, hr_transform: A.Compose = None, lr_transform: A.Compose = None,
            ):
        self.hr_paths = []
        for paths in hr_paths:
            if paths is not None:
                self.hr_paths.append(retrieve_imgs(paths))

        self.lr_paths = self.hr_paths

        self.enlarge_ratio = enlarge_ratio
        self.cutmix_n = cutmix_n
        self.normal_index = normal_index
        self.channels = channels
        self.shared_scale = shared_scale
        self.lr_scale = lr_scale
        self.hr_crop = hr_crop
        self.shared_transform = shared_transform
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

    def getitem(self, index):
        index = index % len(self.hr_paths[0])
        img_hrs = [cv2.imread(paths[index], cv2.IMREAD_UNCHANGED) for paths in self.hr_paths]
        
        last = None
        for img_hr in img_hrs:
            if last is not None:
                if img_hr.shape[0] != last.shape[0] or img_hr.shape[1] != last.shape[1]:
                    raise ValueError("Images must have the same shape")
            last = img_hr

        shared_scale = random.uniform(self.shared_scale[0], self.shared_scale[1])
        if img_hr.shape[1] * shared_scale < self.hr_crop or img_hr.shape[0] * shared_scale < self.hr_crop:
            shared_scale = max(self.hr_crop / img_hr.shape[1], self.hr_crop / img_hr.shape[0])

        img_hrs = [cv2.resize(img_hr, (int(img_hr.shape[1] * shared_scale), int(img_hr.shape[0] * shared_scale)), interpolation=cv2.INTER_AREA) for img_hr in img_hrs]

        # random 90 degree rotations and flip
        rotate = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
        flip = random.choice([None, 0, 1, -1])

        if rotate is not None and flip is not None:
            for i, img_hr in enumerate(img_hrs):
                if rotate is not None:
                    img_hr = cv2.rotate(img_hr, rotate)
                    if i == self.normal_index:
                        img_hr = fix_normal_rotate(img_hr, rotate)
                if flip is not None:
                    img_hr = cv2.flip(img_hr, flip)
                    if i == self.normal_index:
                        img_hr = fix_normal_flip(img_hr, flip)

                img_hrs[i] = img_hr

        if self.shared_transform is not None:
            img_hrs[0] = self.shared_transform(image=img_hrs[0])['image']

        hr_crop_x = random.randint(0, img_hrs[0].shape[1] - self.hr_crop)
        hr_crop_y = random.randint(0, img_hrs[0].shape[0] - self.hr_crop)

        img_hrs = [img_hr[hr_crop_y:hr_crop_y+self.hr_crop, hr_crop_x:hr_crop_x+self.hr_crop] for img_hr in img_hrs]

        interp_mode = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        img_lrs = [cv2.resize(img_hr, (int(img_hr.shape[1] * self.lr_scale), int(img_hr.shape[0] * self.lr_scale)), interpolation=interp_mode) for img_hr in img_hrs]

        if self.hr_transform is not None:
            img_hrs = apply_shared_transform(img_hrs, self.hr_transform)

        if self.lr_transform is not None:
            img_lrs = apply_shared_transform(img_lrs, self.lr_transform)

        img_hrs = [img_to_tensor(img_hr, channels=num_channels) for img_hr, num_channels in zip(img_hrs, self.channels)]
        img_lrs = [img_to_tensor(img_lr, channels=num_channels) for img_lr, num_channels in zip(img_lrs, self.channels)]

        return img_lrs, img_hrs 
    
    def __getitem__(self, index):
        img_lrs, img_hrs = self.getitem(index)

        for _ in range(self.cutmix_n):
            # deterministically randomize the index
            index = hash("index" + str(index)) % len(self.hr_paths[0])
            img_lrs_, img_hrs_ = self.getitem(index)

            # random rectangle between 25% and 50% of the image
            hr_w = random.randint(img_hrs[0].shape[2] // 4, img_hrs[0].shape[2] // 2)
            hr_h = random.randint(img_hrs[0].shape[1] // 4, img_hrs[0].shape[1] // 2)
            hr_x = random.randint(0, img_hrs[0].shape[2] - hr_w)
            hr_y = random.randint(0, img_hrs[0].shape[1] - hr_h)

            scale = img_lrs[0].shape[1] / img_hrs[0].shape[1]
            lr_w = int(hr_w * scale)
            lr_h = int(hr_h * scale)
            lr_x = int(hr_x * scale)
            lr_y = int(hr_y * scale)

            for i in range(len(img_hrs)):
                img_hrs[i][:, hr_y:hr_y+hr_h, hr_x:hr_x+hr_w] = img_hrs_[i][:, hr_y:hr_y+hr_h, hr_x:hr_x+hr_w]
                img_lrs[i][:, lr_y:lr_y+lr_h, lr_x:lr_x+lr_w] = img_lrs_[i][:, lr_y:lr_y+lr_h, lr_x:lr_x+lr_w]

        return img_lrs, img_hrs
    
    def __len__(self):
        return len(self.lr_paths[0]) * self.enlarge_ratio


# VSGAN's tiled super-resolution inference code
# Copyright (c) 2019-2021 PHOENiX
# licenses/VSGAN_LICENSE
# https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/utilities.py

import gc

def tile_tensor(t: torch.Tensor, overlap: int = 16):
    h, w = t.shape[-2:]

    top_left_lr = t[..., : h // 2 + overlap, : w // 2 + overlap]
    top_right_lr = t[..., : h // 2 + overlap, w // 2 - overlap:]
    bottom_left_lr = t[..., h // 2 - overlap:, : w // 2 + overlap]
    bottom_right_lr = t[..., h // 2 - overlap:, w // 2 - overlap:]

    return top_left_lr, top_right_lr, bottom_left_lr, bottom_right_lr

def tiled_forward(
    model: torch.nn.Module | Callable,
    t: torch.Tensor,
    overlap = 16,
    max_depth = None,
    current_depth = 1,
    scale = 1,
    max_tile_size = 1024,
):
    if current_depth > 10:
        torch.cuda.empty_cache()
        gc.collect()
        raise RecursionError("Exceeded maximum tiling recursion of 10...")

    if (max_depth is None or max_depth == current_depth) and (t.size(-2) <= max_tile_size + overlap and t.size(-1) <= max_tile_size + overlap):
        # attempt non-tiled super-resolution if no known depth, or at depth
        try:
            with torch.no_grad():
                t_sr = model(t)
            return t_sr.float().cpu(), current_depth # move to cpu so we aren't hogging VRAM from the other tiles
        except RuntimeError as e:
            if "allocate" in str(e) or "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

    tiles_lr = tile_tensor(t, overlap)
    tiles_lr_top_left, depth = tiled_forward(model, tiles_lr[0], overlap, current_depth=current_depth + 1, scale=scale, max_tile_size=max_tile_size)
    tiles_lr_top_right, _ = tiled_forward(model, tiles_lr[1], overlap, depth, current_depth=current_depth + 1, scale=scale, max_tile_size=max_tile_size)
    tiles_lr_bottom_left, _ = tiled_forward(model, tiles_lr[2], overlap, depth, current_depth=current_depth + 1, scale=scale, max_tile_size=max_tile_size)
    tiles_lr_bottom_right, _ = tiled_forward(model, tiles_lr[3], overlap, depth, current_depth=current_depth + 1, scale=scale, max_tile_size=max_tile_size)

    output_img = join_tiles(
        (tiles_lr_top_left, tiles_lr_top_right, tiles_lr_bottom_left, tiles_lr_bottom_right),
        overlap * scale
    )

    return output_img, depth

def join_tiles(tiles, overlap): 
    h, w = tiles[0].shape[-2:]

    h = (h - overlap) * 2
    w = (w - overlap) * 2

    joined_tile = torch.empty(tiles[0].shape[:-2] + (h, w), dtype=tiles[0].dtype, device=tiles[0].device)
    joined_tile[..., : h // 2, : w // 2] = tiles[0][..., : h // 2, : w // 2]
    joined_tile[..., : h // 2, -w // 2:] = tiles[1][..., : h // 2, -w // 2:]
    joined_tile[..., -h // 2:, : w // 2] = tiles[2][..., -h // 2:, : w // 2]
    joined_tile[..., -h // 2:, -w // 2:] = tiles[3][..., -h // 2:, -w // 2:]

    return joined_tile

# End of VSGAN code