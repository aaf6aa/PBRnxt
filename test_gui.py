import cv2
import dearpygui.dearpygui as dpg
import DearPyGui_DragAndDrop as dpg_dnd
import gc
import imageio
import numpy as np
import os
import random
from spandrel import ImageModelDescriptor, MAIN_REGISTRY, ModelLoader
import torch
import torch.utils.data
from torchvision.transforms.functional import adjust_contrast, adjust_brightness, adjust_saturation

import brdf
from data import img_to_tensor, tensor_to_img, tiled_forward
from pbrnxt_net import PbrNxtNet

if not torch.cuda.is_available():
    print('CUDA is not available. Exiting...')
    exit()

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)

default_device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)
PREVIEW_SIZE = 256
RENDER_SIZE = 512

if torch.cuda.is_bf16_supported():
    default_dtype = torch.bfloat16
else:
    props = torch.cuda.get_device_properties(default_device)
    # fp16 supported at compute 5.3 and above
    if props.major > 5 or (props.major == 5 and props.minor >= 3):
        default_dtype = torch.float16
    else:
        default_dtype = torch.float32

def np_img_to_dpg(img: np.ndarray, size = (PREVIEW_SIZE, PREVIEW_SIZE), bgr2rgb = False):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    max_value = np.iinfo(img.dtype).max
    return (img / max_value).flatten()

def tensor_to_dpg(img: torch.Tensor, size = (PREVIEW_SIZE, PREVIEW_SIZE)):
    img = tensor_to_img(img)
    return np_img_to_dpg(img, size=size, bgr2rgb=True)

class pbrnxtGUI:
    def __init__(self):
        # Default images
        self.lr = img_to_tensor(cv2.imread("examples/example_lr.png", cv2.IMREAD_COLOR)).unsqueeze(0).to(torch.device("cpu"), default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
        self.sr_diffuse = torch.ones((1, 3, PREVIEW_SIZE, PREVIEW_SIZE)).to(torch.device("cpu"), default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
        self.sr_normal = torch.ones((1, 3, PREVIEW_SIZE, PREVIEW_SIZE)).to(torch.device("cpu"), default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
        self.sr_rough = torch.ones((1, 1, PREVIEW_SIZE, PREVIEW_SIZE)).to(torch.device("cpu"), default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
        self.sr_disp = torch.ones((1, 1, PREVIEW_SIZE, PREVIEW_SIZE)).to(torch.device("cpu"), default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
        self.alpha = None
        self.render = self.sr_diffuse

        self.preview_lr = self.lr
        self.preview_sr_diffuse = self.sr_diffuse
        self.preview_sr_normal = self.sr_normal
        self.preview_sr_rough = self.sr_rough
        self.preview_sr_disp = self.sr_disp

        # DearPyGui setup
        dpg.create_context()
        dpg_dnd.initialize()

        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=PREVIEW_SIZE, height=PREVIEW_SIZE, default_value=tensor_to_dpg(self.preview_lr), tag="in_diffuse_img")
            dpg.add_dynamic_texture(width=PREVIEW_SIZE, height=PREVIEW_SIZE, default_value=tensor_to_dpg(self.preview_sr_diffuse), tag="out_diffuse_img")
            dpg.add_dynamic_texture(width=PREVIEW_SIZE, height=PREVIEW_SIZE, default_value=tensor_to_dpg(self.preview_sr_normal), tag="out_normal_img")
            dpg.add_dynamic_texture(width=PREVIEW_SIZE, height=PREVIEW_SIZE, default_value=tensor_to_dpg(self.preview_sr_rough), tag="out_rough_img")
            dpg.add_dynamic_texture(width=PREVIEW_SIZE, height=PREVIEW_SIZE, default_value=tensor_to_dpg(self.preview_sr_disp), tag="out_disp_img")
            dpg.add_dynamic_texture(width=PREVIEW_SIZE, height=PREVIEW_SIZE, default_value=tensor_to_dpg(self.render), tag="render_img")

        dpg.create_viewport(title="pbrnxt GUI", width=1184, height=736, clear_color=(64, 64, 64, 255))
        dpg.setup_dearpygui()

        # Input diffuse
        with dpg.window(label="Diffuse Input", id="in_preview", pos=[16, 16], no_close=True, no_collapse=True):
            dpg.add_image("in_diffuse_img")
            dpg.add_text("Resolution: 256x256", tag="in_resolution")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Open Texture", callback=lambda: dpg.show_item("in_dialog"))
                dpg.add_button(label="Generate", callback=self.process, tag="process_button")
        
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.update_in_diffuse, id="in_dialog", width=700 ,height=400):
            dpg.add_file_extension("Texture{.png,.dds,.jpg,.jpeg,.tif,.tiff,.tga,.bmp}")
            dpg.add_file_extension(".*")

        # Drag and Drop image or model
        def drop(data, _):
            filepath = data[0]
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ['.pt', '.pth', '.safetensors', '.ckpt']:
                self.load_custom_upscaler(None, {'file_path_name': filepath}, None)
            else:
                self.update_in_diffuse(None, {'file_path_name': filepath}, None)
        dpg_dnd.set_drop(drop)

        with dpg.window(label="Processing . . .", pos=[256, 256], width=200, height=24, modal=True, show=False, tag="process_popup", no_title_bar=True):
            dpg.add_text("Processing . . .", tag="progress_text")

        # Custom upscaler
        with dpg.window(label="Custom Diffuse Upscaler", id="upscale_win", pos=[292, 620], width=536, height=24, show=True, no_close=True, no_collapse=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Model", callback=lambda: dpg.show_item("model_dialog"))
                dpg.add_button(label="Reset", callback=lambda: (setattr(self, "custom_upscaler", None), dpg.set_value("custom_upscaler_name", "None")))
                dpg.add_text("None", tag="custom_upscaler_name")

        with dpg.file_dialog(directory_selector=False, show=False, callback=self.load_custom_upscaler, id="model_dialog", width=700 ,height=400):
            dpg.add_file_extension("PyTorch Model{.pt,.pth,.safetensors,.ckpt}")
            dpg.add_file_extension(".*")

        # Output preview
        with dpg.window(label="Material Preview", id="out_preview", pos=[292, 16], show=False, no_close=True, no_collapse=True):
            with dpg.group(horizontal=False):
                with dpg.group(horizontal=True):
                    dpg.add_image("out_diffuse_img")
                    dpg.add_image("out_normal_img")
                with dpg.group(horizontal=True):
                    dpg.add_image("out_rough_img")
                    dpg.add_image("out_disp_img")
            dpg.add_text("Resolution: 256x256", tag="out_resolution")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=lambda: dpg.show_item("out_dialog"))
                dpg.add_input_text(label="Material Name", tag="material_name", default_value="example_material", width=256)

            dpg.add_file_dialog(label="Material Directory", directory_selector=True, show=False, callback=self.save_material, tag="out_dialog", width=700 ,height=400)
        
        # Material properties
        with dpg.window(label='Diffuse', id='diffuse_win', pos=[832, 16], width=320, height=144, show=False):
            dpg.add_slider_float(label='Brightness', tag="diffuse_brightness", default_value=1.0, min_value=0.0, max_value=3.0, callback=self.update_diffuse)
            dpg.add_slider_float(label='Contrast', tag="diffuse_contrast", default_value=1.0, min_value=0.0, max_value=4.0, callback=self.update_diffuse)
            dpg.add_slider_float(label='Saturation', tag="diffuse_saturation", default_value=1.0, min_value=0.0, max_value=2.0, callback=self.update_diffuse)
            dpg.add_checkbox(label='De-Light', tag="diffuse_delight", default_value=False, callback=self.update_diffuse)

            dpg.add_button(label="Reset", callback=lambda: (dpg.set_value("diffuse_brightness", 1.0), dpg.set_value("diffuse_contrast", 1.0), dpg.set_value("diffuse_saturation", 1.0), dpg.set_value("diffuse_delight", False), self.update_diffuse()))

        with dpg.window(label='Normal', id='normal_win', pos=[832, 160], width=320, height=144, show=False):
            dpg.add_checkbox(label='Invert', tag="normal_invert", callback=self.update_normal)

            dpg.add_button(label="Reset", callback=lambda: (dpg.set_value("normal_invert", False), self.update_normal()))

        with dpg.window(label='Roughness', id='rough_win', pos=[832, 304], width=320, height=144, show=False):
            dpg.add_slider_float(label='Contrast', tag="rough_contrast", default_value=1.0, min_value=0.0, max_value=4.0, callback=self.update_rough)
            dpg.add_slider_float(label='Bias', tag="rough_bias", default_value=0.0, min_value=-1.0, max_value=1.0, callback=self.update_rough)
            dpg.add_checkbox(label='Invert', tag="rough_invert", callback=self.update_rough)

            dpg.add_button(label="Reset", callback=lambda: (dpg.set_value("rough_contrast", 1.0), dpg.set_value("rough_bias", 0.0), dpg.set_value("rough_invert", False), self.update_rough()))
        
        with dpg.window(label='Displacement', id='disp_win', pos=[832, 448], width=320, height=160, show=False):
            dpg.add_slider_float(label='Contrast', tag="disp_contrast", default_value=1.0, min_value=0.0, max_value=4.0, callback=self.update_disp)
            dpg.add_slider_float(label='Bias', tag="disp_bias", default_value=0.0, min_value=-1.0, max_value=1.0, callback=self.update_disp)
            dpg.add_checkbox(label='Invert', tag="disp_invert", callback=self.update_disp)

            dpg.add_button(label="Reset", callback=lambda: (dpg.set_value("disp_contrast", 1.0), dpg.set_value("disp_bias", 0.0), dpg.set_value("disp_invert", False), self.update_disp()))
            dpg.add_text("Note: Displacement maps are not\nsupported in the renderer.")

        # Render preview
        self.renderer = brdf.Renderer(default_device)
        with dpg.window(label="Material Render", id="render_preview", pos=[16, 352], show=False, no_close=True, no_collapse=True):
            dpg.add_image("render_img")
            dpg.add_button(label="Randomize Light", callback=lambda: self.update_render(True))

        with dpg.window(label="Performance", id="perf_win", pos=[832, 620], width=320, height=24, show=False, no_close=True, no_collapse=True):
            dpg.add_text("Execution time: 0s", tag="perf_time")
            dpg.add_text("VRAM used: 0MB", tag="perf_vram")

        # Network initialization
        self.network_name = "pbrnxt"
        self.out_channels = [3, 3, 1, 1]
        self.scale = 4
        
        self.network = PbrNxtNet(3, self.out_channels, 96, 32, self.scale, 512)
        self.network = self.network.to(torch.device("cpu"), default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
        self.network.eval()
        self.network.load_state_dict(torch.load(f"pretrained_models/pbrnxt_402236.pth"), strict=True)
        self.custom_upscaler = None
    
    def update_in_diffuse(self, sender, app_data, user_data):
        if not app_data:
            return
        if 'file_path_name' not in app_data or not app_data['file_path_name']:
            return
        
        img_path = app_data['file_path_name']
        if not os.path.exists(img_path) or not os.path.isfile(img_path):
            return
        
        material_name = os.path.basename(img_path).split('.')[0]
        dpg.set_value('material_name', material_name)
        
        if img_path.endswith(".dds"):
            img = imageio.v2.imread(img_path, format='DDS')
            bgr2rgb = False
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            bgr2rgb = True

        self.lr = img_to_tensor(img, bgr2rgb=bgr2rgb, channels=3).unsqueeze(0).to(torch.device("cpu"), default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
        if img.ndim == 3 and img.shape[2] == 4:
            self.alpha = img_to_tensor(img[:, :, 3], channels=1).unsqueeze(0).to(torch.device("cpu"), default_dtype, non_blocking=True).to(memory_format=torch.channels_last)
            self.lr = torch.cat([self.lr, self.alpha], dim=1)
        else:
            self.alpha = None

        self.preview_lr = self.lr

        dpg.set_value('in_diffuse_img', tensor_to_dpg(self.preview_lr))
        dpg.set_value('in_resolution', f"Resolution: {self.lr.size(3)}x{self.lr.size(2)}")

    def load_custom_upscaler(self, sender, app_data, user_data):
        if not app_data:
            return
        if 'file_path_name' not in app_data or not app_data['file_path_name']:
            return
        
        model_path = app_data['file_path_name']
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            return

        model_name = os.path.basename(model_path).split('.')[0]
        model = ModelLoader().load_from_file(model_path)
        if not isinstance(model, ImageModelDescriptor):
            print(f"{model_name} is not an image-to-image model.")
            return

        if model.input_channels != 3 or model.output_channels != 3:
            print(f"{model_name} must have 3 input and output channels.")
            return

        self.custom_upscaler = model.to(torch.device("cpu"), default_dtype).eval()
        dpg.set_value('custom_upscaler_name', model_name)

    def process(self):
        dpg.configure_item("process_popup", show=True)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        vram_used = 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(default_dtype is not torch.float32), dtype=default_dtype):
            start.record()
            lr = self.lr[:, :3, ...].to(default_device)

            dpg.set_value("progress_text", "Generating PBR maps . . .")
            torch.cuda.empty_cache()
            gc.collect()

            self.network = self.network.to(default_device)
            img_srs, _ = tiled_forward(self.network, lr, overlap=64, scale=self.scale, max_tile_size=512)
            vram_used = max(vram_used, torch.cuda.max_memory_allocated(device=None))

            self.network = self.network.to(torch.device("cpu"))
            self.lr = self.lr.to(torch.device("cpu"))
            img_srs = img_srs.to(torch.device("cpu"))
            img_srs = list(torch.split(img_srs, self.out_channels, dim=1))

            # custom upscaler
            if self.custom_upscaler is not None:
                dpg.set_value("progress_text", "Applying custom upscaler . . .")
                torch.cuda.empty_cache()
                gc.collect()

                self.custom_upscaler = self.custom_upscaler.to(default_device)

                diffuse = img_srs[0].to(default_device)
                diffuse = torch.nn.functional.interpolate(diffuse, (self.lr.size(2), self.lr.size(3)), mode='bilinear')
                diffuse, _ = tiled_forward(self.custom_upscaler, diffuse, overlap=64, scale=self.custom_upscaler.scale, max_tile_size=1024)
                vram_used = max(vram_used, torch.cuda.max_memory_allocated(device=None))

                img_srs[0] = diffuse.to(torch.device("cpu"))
                self.custom_upscaler = self.custom_upscaler.to(torch.device("cpu"))

            end.record()
        
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000.0
        dpg.set_value("perf_time", f"Execution time: {elapsed_time:.2f}s")
        dpg.set_value("perf_vram", f"VRAM used: {vram_used/1024/1024:.2f}MB")
        
        self.sr_diffuse = img_srs[0]
        if self.sr_diffuse.size(1) != 4 and self.alpha is not None:
            alpha = torch.nn.functional.interpolate(self.alpha, (self.sr_diffuse.size(2), self.sr_diffuse.size(3)), mode='bilinear')
            self.sr_diffuse = torch.cat([self.sr_diffuse, alpha], dim=1)
        self.sr_normal = img_srs[1]
        self.sr_rough = img_srs[2]
        self.sr_disp = img_srs[3]

        self.update_diffuse()
        self.update_normal()
        self.update_rough()
        self.update_disp()

        self.update_preview()
        self.update_render()

        dpg.configure_item("process_popup", show=False)
        
    def update_diffuse(self):
        diffuse = self.sr_diffuse

        # torchvision transforms do not support alpha channel
        if self.sr_diffuse.size(1) == 4:
            alpha = diffuse[:, 3:4]
            diffuse = diffuse[:, :3]

        if dpg.get_value("diffuse_delight"):
            diffuse = diffuse * (1.0 - adjust_saturation(diffuse, 0.0))

        brightness = dpg.get_value("diffuse_brightness")
        contrast = dpg.get_value("diffuse_contrast")
        saturation = dpg.get_value("diffuse_saturation")

        diffuse = adjust_brightness(diffuse, brightness)
        diffuse = adjust_contrast(diffuse, contrast)
        diffuse = adjust_saturation(diffuse, saturation)

        if self.sr_diffuse.size(1) == 4:
            diffuse = torch.cat([diffuse, alpha], dim=1)

        self.preview_sr_diffuse = diffuse

        dpg.set_value('out_diffuse_img', tensor_to_dpg(self.preview_sr_diffuse))
        self.update_render()

    def update_normal(self):
        normal = self.sr_normal.clone()

        if dpg.get_value("normal_invert"):
            normal[:, 1] = 1.0 - normal[:, 1]

        # normalize
        normal = (normal * 2.0 - 1.0)
        normal = normal / torch.norm(normal, dim=1, keepdim=True)
        normal = (normal + 1.0) / 2.0

        self.preview_sr_normal = normal
        dpg.set_value('out_normal_img', tensor_to_dpg(self.preview_sr_normal))
        self.update_render()
    
    def update_rough(self):
        rough = self.sr_rough.clone()

        contrast = dpg.get_value("rough_contrast")
        bias = dpg.get_value("rough_bias")
        rough = adjust_contrast(rough, contrast)
        rough = rough + bias

        rough = rough.clamp(0.0, 1.0)
        if dpg.get_value("rough_invert"):
            rough = 1.0 - rough
        
        self.preview_sr_rough = rough
        dpg.set_value('out_rough_img', tensor_to_dpg(self.preview_sr_rough))
        self.update_render()
    
    def update_disp(self):
        disp = self.sr_disp.clone()

        contrast = dpg.get_value("disp_contrast")
        bias = dpg.get_value("disp_bias")
        disp = adjust_contrast(disp, contrast)
        disp = disp + bias

        # set displacement mean to 0.5
        std, mean = torch.std_mean(disp)
        new_std = std * (0.5 / mean)
        disp = (disp - mean) / std * new_std + 0.5

        disp = disp.clamp(0.0, 1.0)
        if dpg.get_value("disp_invert"):
            disp = 1.0 - disp

        self.preview_sr_disp = disp
        dpg.set_value('out_disp_img', tensor_to_dpg(self.preview_sr_disp))
        self.update_render()

    def update_preview(self):
        dpg.set_value('out_diffuse_img', tensor_to_dpg(self.preview_sr_diffuse))
        dpg.set_value('out_normal_img', tensor_to_dpg(self.preview_sr_normal))
        dpg.set_value('out_rough_img', tensor_to_dpg(self.preview_sr_rough))
        dpg.set_value('out_disp_img', tensor_to_dpg(self.preview_sr_disp))
        dpg.set_value('out_resolution', f"Resolution: {self.sr_diffuse.size(3)}x{self.sr_diffuse.size(2)}")
        
        dpg.show_item("out_preview")
        dpg.show_item("perf_win")

        dpg.show_item("diffuse_win")
        dpg.show_item("normal_win")
        dpg.show_item("rough_win")
        dpg.show_item("disp_win")

    def update_render(self, step=False):
        if step:
            self.renderer.step()

        a = torch.nn.functional.interpolate(self.preview_sr_diffuse, (RENDER_SIZE, RENDER_SIZE), mode='bilinear', align_corners=False).to(default_device)
        if self.sr_diffuse.size(1) == 4:
            alpha = a[:, 3:4]
            a = a[:, :3]

        n = torch.nn.functional.interpolate(self.preview_sr_normal, (RENDER_SIZE, RENDER_SIZE), mode='bilinear', align_corners=False).to(default_device)
        r = torch.nn.functional.interpolate(self.preview_sr_rough, (RENDER_SIZE, RENDER_SIZE), mode='bilinear', align_corners=False).to(default_device)
        d = torch.nn.functional.interpolate(self.preview_sr_disp, (RENDER_SIZE, RENDER_SIZE), mode='bilinear', align_corners=False).to(default_device)
        self.render = self.renderer.render(a, n, r, d)

        if self.sr_diffuse.size(1) == 4:
            self.render = torch.cat([self.render, alpha], dim=1)
        
        self.render = self.render.to(torch.device("cpu"))

        dpg.set_value('render_img', tensor_to_dpg(self.render))
        dpg.show_item("render_preview")

    def save_material(self, sender, app_data, user_data):
        if not app_data:
            return
        if 'file_path_name' not in app_data or not app_data['file_path_name']:
            return
        
        dir_path = app_data['file_path_name']
        os.makedirs(dir_path, exist_ok=True)
        material_name = dpg.get_value('material_name') + "_" + self.network_name
        if self.custom_upscaler is not None:
            material_name += "_" + dpg.get_value('custom_upscaler_name')

        diffuse = tensor_to_img(self.preview_sr_diffuse)
        cv2.imwrite(os.path.join(dir_path, f"{material_name}_albedo.png"), diffuse)

        normal = tensor_to_img(self.preview_sr_normal)
        cv2.imwrite(os.path.join(dir_path, f"{material_name}_normal_gl.png"), normal)
        # flip green channel for directx normal map
        normal[:, :, 1] = 65535 - normal[:, :, 1]
        cv2.imwrite(os.path.join(dir_path, f"{material_name}_normal_dx.png"), normal)

        roughness = tensor_to_img(self.preview_sr_rough)
        cv2.imwrite(os.path.join(dir_path, f"{material_name}_roughness.png"), cv2.cvtColor(roughness, cv2.COLOR_BGR2GRAY))

        displacement = tensor_to_img(self.preview_sr_disp)
        cv2.imwrite(os.path.join(dir_path, f"{material_name}_displacement.png"), cv2.cvtColor(displacement, cv2.COLOR_BGR2GRAY))

    def run(self):
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
        return


if __name__ == "__main__":
    gui = pbrnxtGUI()
    gui.run()
