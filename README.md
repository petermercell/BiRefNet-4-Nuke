# BiRefNet_Portrait for Nuke

A TensorRT-accelerated **BiRefNet portrait matting** node for Foundry Nuke.

Drop in an RGB plate, get back RGBA with a high-quality portrait matte in the
alpha channel. The TensorRT engine is **embedded directly in the `.so`** —
no external model files, no Cattery, no Python. The only runtime dependency
is the NVIDIA driver.

> Built on top of the excellent [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
> by Peng Zheng et al. (MIT license). All credit for the model architecture
> and weights belongs to them.

---

## Features

- **Pure NDK C++ plugin** — runs natively in Nuke's image graph as `AI/BiRefNet_Portrait`.
- **TensorRT 10.x inference** via `enqueueV3` and named tensors.
- **Self-contained binary** — TensorRT, cuDNN-free TRT static libs, cuBLAS,
  NVRTC and CUDA runtime are statically linked. The serialized engine
  is `objcopy`'d into a `.rodata` section of the plugin itself.
- **Linear-workflow safe** — RGB passthrough stays in your Nuke working space;
  only the model input is converted to sRGB and ImageNet-normalized for inference.
- **Sigmoid applied internally** — the engine outputs raw logits, the plugin
  takes care of the activation.
- **Multi-threaded scanline output** — Nuke's per-row `engine()` is fully
  supported; the inference itself runs once per frame behind a mutex,
  results are then read out per scanline.
- **Frame-coherent caching** — inference is triggered lazily on the first row
  request and reused for the rest of the frame.

## Requirements

### Runtime (target machine)

- Linux x86_64
- NVIDIA driver compatible with **CUDA 12.8** (R555+)
- Foundry Nuke 14.1+ NDK ABI (built and tested on **Nuke 17.0v1**)
- An NVIDIA GPU with enough VRAM for the engine (~1–2 GB for FP32 BiRefNet portrait)

That's it — no TensorRT install required on the target machine.

### Build (developer machine)

- Linux x86_64, GCC 9+ with C++17 support
- CMake 3.18+
- CUDA Toolkit 12.8
- TensorRT 10.9.0.34 (with **static** libraries — `libnvinfer_static.a`)
- Nuke 17.0v1 (or matching NDK)
- A pre-built TensorRT engine file (see below)

## Building from source

### 1. Generate the TensorRT engine from the BiRefNet ONNX

Grab the portrait ONNX from the
[BiRefNet repo](https://github.com/ZhengPeng7/BiRefNet) (or convert the
PyTorch checkpoint yourself), then build the engine with `trtexec`:

```bash
export LD_LIBRARY_PATH=/opt/TensorRT-10.9.0.34/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

/opt/TensorRT-10.9.0.34/bin/trtexec \
    --onnx=/path/to/BiRefNet-portrait-epoch_150.onnx \
    --saveEngine=/path/to/BiRefNet-portrait-epoch_150_fp32_cuda128.engine \
    --memPoolSize=workspace:4G
```

> The HR matting variant works the same way — just swap the ONNX path
> and rename the output engine. See `GUIDE.txt` for the exact command
> used for the HR variant.

### 2. Place the engine and build

Drop the resulting `.engine` next to `CMakeLists.txt` and name it
`BiRefNet-portrait-epoch_150_fp32_cuda128.engine` (or override
`-DENGINE_FILE=...` on the CMake line):

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

rm -rf build && mkdir build && cd build
cmake .. \
    -DNUKE_ROOT=/opt/Nuke17.0v1 \
    -DTENSORRT_ROOT=/opt/TensorRT-10.9.0.34
make -j8
```

The build will produce a single self-contained `BiRefNet_Portrait.so`
in the build directory.

## Usage in Nuke

Pipe an RGB plate in, attach a `BiRefNet_Portrait`, and read the alpha
channel.

```
Read → BiRefNet_Portrait → Premult / Copy / Merge
```

### Knobs

| Knob | Description |
|---|---|
| **Matte Only (BW)** | Replaces the RGB passthrough with the matte rendered as greyscale. Useful for QC and for chaining into despill / refinement nodes. |
| **Invert Matte** | Inverts the alpha — handy if you want a background matte instead of the foreground. |
| **Input is Linear** | Enabled by default. Applies a linear → sRGB conversion to the *model input only* before inference, because BiRefNet was trained on gamma-encoded images. The RGB passthrough and the alpha output are unaffected, so your linear workflow is preserved. Disable if your input is already sRGB / Rec.709 encoded. |

## Technical notes

### Model

| | |
|---|---|
| Input tensor   | `input_image`, `float32[1, 3, 1024, 1024]` (NCHW) |
| Output tensor  | `output_image`, `float32[1, 1, 1024, 1024]` (NCHW), raw logits |
| Activation     | Sigmoid (applied by the plugin) |
| Normalization  | ImageNet mean/std on the model input, sRGB-encoded |

Tensor names are confirmed via Netron and `trtexec --verbose`.

### Pipeline per frame

1. Fetch full-resolution RGB into a planar CPU buffer.
2. Bilinear resize to **1024 × 1024**.
3. Optional linear → sRGB on the resized model input.
4. ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
5. `cudaMemcpyAsync` → `enqueueV3` → `cudaMemcpyAsync` → `cudaStreamSynchronize`.
6. Sigmoid on the output logits.
7. Bilinear resize back to source resolution.
8. Optional invert.
9. Per-scanline output: original RGB (or matte-as-grey) + matte in alpha.

### Engine embedding

`CMakeLists.txt` invokes `objcopy` to wrap the `.engine` file as an ELF
object with the symbols `_binary_engine_bin_start` and `_binary_engine_bin_end`,
which are then read at `_open()` time and handed to
`IRuntime::deserializeCudaEngine()`. This keeps deployment to a single file.

### Static linking

The plugin links statically against:

- `libnvinfer_static.a` (TensorRT)
- `libcudart_static.a`, `libcublas_static.a`, `libcublasLt_static.a`
- `libnvrtc_static.a`, `libnvrtc-builtins_static.a`
- `libnvptxcompiler_static.a`, `libnvJitLink_static.a`
- `libculibos.a`

…and dynamically against `libDDImage.so`, `libdl`, `librt`, `libpthread`,
plus the NVIDIA driver's `libcuda.so`. There are no other shared dependencies.

## A note on the HR matting variant

A separate node for the higher-resolution `BiRefNet_HR-matting` model can
be built with the same source by swapping the embedded engine. The
`GUIDE.txt` file in this repo includes the exact `trtexec` command used
for the HR variant.

## Credits

- **BiRefNet** — model architecture and weights by **Peng Zheng et al.**
  Released under the MIT license at
  [github.com/ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
  Thank you for the outstanding work and for open-sourcing it.
- **TensorRT** — NVIDIA Corporation.
- **Nuke NDK** — The Foundry Visionmongers Ltd.

## License

Released under the **MIT License**. See `LICENSE` for details.

The embedded BiRefNet engine is derived from weights distributed under the
MIT license; the original copyright belongs to the BiRefNet authors.

## Author

**Peter Mercell** — independent VFX developer, Prague.

- Website: [petermercell.com](https://petermercell.com)
- Patreon: [patreon.com/cw/PeterMercell](https://www.patreon.com/cw/PeterMercell)

If you find this useful and want to support more open-source Nuke tools,
the Patreon is the place. Bug reports and PRs are welcome here on GitHub.
