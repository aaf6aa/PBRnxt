# PBRnxt
A deep learning based tool for generating high-resolution PBR materials from flat diffuse textures. This generates an albedo, DirectX and OpenGL normal maps, a roughness map, and a displacement map. The tool consists of a custom hybrid transformer-convnet architecture based on [SCUNet](https://github.com/cszn/SCUNet), trained on public domain and CC0 licensed PBR materials.

Paper detailing the design will be released in the future. Designed and developed in partial fulfillment of a Computer Science Integrated Masters degree.

![GUI Screenshot](https://github.com/aaf6aa/PBRnxt/assets/56702415/401347cc-8b05-4fe9-bff4-484087712822)

## Showcase

Comparisons between 256x256 input textures and PBR materials generated from them and rendered in Blender Cycles.

![bricks_256px_comp](https://github.com/aaf6aa/PBRnxt/assets/56702415/3a026902-6088-45a0-9666-b6e4131cbafd)
![ground_256px_comp](https://github.com/aaf6aa/PBRnxt/assets/56702415/cb03a3b3-cc55-4994-97ec-9fdf85f09d7d)
![paving_256px_comp](https://github.com/aaf6aa/PBRnxt/assets/56702415/848d2733-4309-4094-8eed-a05a468749c4)

https://github.com/aaf6aa/PBRnxt/assets/56702415/a375c1cb-1627-4214-81ea-b9afa335cc54

### Performance

On an RTX 3070, the network achieves an average runtime of 5.1 seconds for 512x512 input textures, compared to ~150 seconds for [NVIDIA's AI Texture tools](https://docs.omniverse.nvidia.com/kit/docs/rtx_remix/latest/docs/howto/learning-aitexturetools.html) included with RTX Remix.

## Prerequisites 
Developed for Python 3.11, PyTorch 2.2, CUDA 11.6+/12.x. Tested only on Windows.

Ensure that one of the either CUDA versions and the latest NVIDIA drivers are installed. All other dependencies can be installed using the corresponding `requirements_cudaXX.txt`.

## Usage
The GUI can be started via either `python test_gui.py` or `test_gui.bat`.

Input files can be selected from a file dialog in the GUI or drag & dropped into the GUI. All common image formats and the DDS texture format are supported, and with and without an alpha channel. Generated material can be saved using a folder dialogue in PNG format. 

### Custom Diffuse Upscaler
An additional network can be specified to upscale the diffuse texture alongside our PBR generation network to potentially further enhance the result, as long as the model is support by [Spandrel](https://github.com/chaiNNer-org/spandrel). Community texture super-resolution models can be found at [OpenModelDB](https://openmodeldb.info/?t=game-textures).

## Acknowledgements
- [AmbientCG](https://ambientcg.com): CC0 PBR materials for training
- [cgbookcase](https://www.cgbookcase.com): CC0 PBR materials for training
- [Poly Haven](https://polyhaven.com): CC0 PBR materials for training
- [SCUNet](https://github.com/cszn/SCUNet): Main basis for the PBR generator network architecture
- [Swin Transformer V2](https://github.com/microsoft/Swin-Transformer): SwinV2 modification applied to SCUNet
- [ESRGAN+](https://github.com/ncarraz/ESRGANplus): Basis for the PBR super-resolution component
- [Dear PyGui](https://github.com/hoffstadt/DearPyGui): GUI framework
- [Crash Course in BRDF Implementation](https://boksajak.github.io/blog/BRDF): BRDF implementation for material renderer
- [Spandrel](https://github.com/chaiNNer-org/spandrel): Support for a wide range of customer upscalers
- [DISTS](https://github.com/dingkeyan93/DISTS): Perceptual loss
- [Textile](https://github.com/crp94/textile): Tiling loss
- [A-ESRGAN](https://github.com/stroking-fishes-ml-corp/A-ESRGAN): Discriminator architecture
- [VSGAN](https://github.com/rlaphoenix/VSGAN): Tiled inference
