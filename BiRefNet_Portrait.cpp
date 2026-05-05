/*
 * BiRefNet_Portrait - TensorRT BiRefNet Portrait Matting Node for Nuke
 *
 * Model details (confirmed via Netron / trtexec):
 *   Input:  "input_image"  float32[1,3,1024,1024]  (NCHW)
 *   Output: "output_image" float32[1,1,1024,1024]  (NCHW)
 *   Last op: Conv (decoder.conv_out1.conv_out1.0.Conv) — raw logits
 *   Sigmoid applied by this node.
 *
 * Built for: TensorRT 10.9.0 (enqueueV3 + named tensors)
 *            Nuke 14.1 NDK
 *            CUDA 12.8
 *            Engine embedded in binary — no external files needed.
 *
 * Author: Peter Mercell
 * Website: petermercell.com
 *
 * BiRefNet model by Peng Zheng et al.
 * https://github.com/ZhengPeng7/BiRefNet
 * Licensed under MIT — thank you for the outstanding work
 * and for open-sourcing BiRefNet!
 */

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Thread.h"
#include "DDImage/Format.h"
#include "DDImage/Interest.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <iostream>
#include <mutex>

using namespace DD::Image;

// ---------------------------------------------------------------------------
// Embedded engine data (linked via objcopy from engine.bin)
// ---------------------------------------------------------------------------
extern "C" {
    extern const char _binary_engine_bin_start[];
    extern const char _binary_engine_bin_end[];
}

// ---------------------------------------------------------------------------
// TensorRT logger
// ---------------------------------------------------------------------------
class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[BiRefNet_Portrait] " << msg << std::endl;
    }
};

static TRTLogger gLogger;

// ---------------------------------------------------------------------------
// Tensor names (confirmed from Netron / trtexec)
// ---------------------------------------------------------------------------
static const char* kInputTensorName  = "input_image";
static const char* kOutputTensorName = "output_image";

// ---------------------------------------------------------------------------
// Fixed model resolution
// ---------------------------------------------------------------------------
static const int kModelW = 1024;
static const int kModelH = 1024;

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            error("CUDA error: %s at %s:%d", cudaGetErrorString(err),           \
                  __FILE__, __LINE__);                                           \
            return;                                                             \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Bilinear resize (CPU, planar)
// ---------------------------------------------------------------------------
static void bilinearResize(const float* src, int srcW, int srcH,
                           float* dst, int dstW, int dstH,
                           int channels)
{
    for (int c = 0; c < channels; ++c)
    {
        const float* srcC = src + c * srcW * srcH;
        float* dstC       = dst + c * dstW * dstH;

        for (int dy = 0; dy < dstH; ++dy)
        {
            float sy = (dy + 0.5f) * srcH / (float)dstH - 0.5f;
            int y0   = (int)std::floor(sy);
            int y1   = y0 + 1;
            float fy = sy - y0;

            y0 = std::max(0, std::min(y0, srcH - 1));
            y1 = std::max(0, std::min(y1, srcH - 1));

            for (int dx = 0; dx < dstW; ++dx)
            {
                float sx = (dx + 0.5f) * srcW / (float)dstW - 0.5f;
                int x0   = (int)std::floor(sx);
                int x1   = x0 + 1;
                float fx = sx - x0;

                x0 = std::max(0, std::min(x0, srcW - 1));
                x1 = std::max(0, std::min(x1, srcW - 1));

                float v00 = srcC[y0 * srcW + x0];
                float v10 = srcC[y0 * srcW + x1];
                float v01 = srcC[y1 * srcW + x0];
                float v11 = srcC[y1 * srcW + x1];

                dstC[dy * dstW + dx] =
                    (1 - fy) * ((1 - fx) * v00 + fx * v10)
                  +      fy  * ((1 - fx) * v01 + fx * v11);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BiRefNet_Portrait node
// ---------------------------------------------------------------------------
class BiRefNet_Portrait : public Iop
{

public:

    BiRefNet_Portrait(Node* node);
    ~BiRefNet_Portrait() override;

    const char* Class() const override { return description.name; }
    const char* node_help() const override
    {
        return "BiRefNet Portrait Matting — portrait foreground matting.\n\n"
               "Runs a TensorRT engine of BiRefNet to produce\n"
               "a foreground matte in the alpha channel.\n\n"
               "Input: RGB image (any resolution)\n"
               "Output: RGBA (alpha = BiRefNet matte)\n\n"
               "Model resolution: 1024x1024\n"
               "Engine is embedded — no external files needed.\n\n"
               "BiRefNet_Portrait for Nuke by Peter Mercell, 2026\n"
               "petermercell.com\n\n"
               "BiRefNet model by Peng Zheng et al.\n"
               "https://github.com/ZhengPeng7/BiRefNet\n"
               "Licensed under MIT.\n"
               "Thank you for the outstanding work and for\n"
               "open-sourcing BiRefNet!";
    }

    void knobs(Knob_Callback f) override;

    void _validate(bool for_real) override;
    void _request(int x, int y, int r, int t,
                  ChannelMask channels, int count) override;
    void _open() override;
    void _close() override;
    void engine(int y, int x, int r,
                ChannelMask channels, Row& row) override;

    static const Iop::Description description;

private:

    // --- knobs ---
    bool        outputMatteOnly_;
    bool        invertMatte_;
    bool        linearInput_;      // Nuke native linear → convert to sRGB before inference
    bool        useImageNetNorm_;
    int         gpuDevice_;

    // --- frame geometry ---
    int frameW_;
    int frameH_;
    int frameX_;
    int frameY_;

    // --- CPU buffers ---
    std::vector<float> cpuFrameIn_;
    std::vector<float> cpuMatteOut_;
    std::vector<float> modelInput_;
    std::vector<float> modelOutput_;

    // --- CUDA ---
    float*       d_input_;
    float*       d_output_;
    cudaStream_t stream_;

    // --- TensorRT 10.x ---
    nvinfer1::IRuntime*          runtime_;
    nvinfer1::ICudaEngine*       engineTRT_;
    nvinfer1::IExecutionContext*  context_;

    // --- frame-level inference lock ---
    std::mutex inferenceMutex_;
    bool       inferenceRan_;

    // --- engine state ---
    bool engineLoaded_;

    // --- methods ---
    void loadEngine();
    void freeEngine();
    void allocateGPU();
    void freeGPU();
    void fetchAllRows();
    void preprocessFrame();
    void runInference();
    void postprocessMatte();
    void doFullInference();
};

// ---------------------------------------------------------------------------
// Ctor / Dtor
// ---------------------------------------------------------------------------
BiRefNet_Portrait::BiRefNet_Portrait(Node* node)
    : Iop(node)
    , outputMatteOnly_(false)
    , invertMatte_(false)
    , linearInput_(true)
    , useImageNetNorm_(true)
    , gpuDevice_(0)
    , frameW_(0), frameH_(0), frameX_(0), frameY_(0)
    , d_input_(nullptr), d_output_(nullptr), stream_(nullptr)
    , runtime_(nullptr), engineTRT_(nullptr), context_(nullptr)
    , inferenceRan_(false)
    , engineLoaded_(false)
{
}

BiRefNet_Portrait::~BiRefNet_Portrait()
{
    freeGPU();
    freeEngine();
}

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::knobs(Knob_Callback f)
{
    Bool_knob(f, &outputMatteOnly_, "matte_only", "Matte Only (BW)");
    Tooltip(f, "Output matte as greyscale RGB.");

    Bool_knob(f, &invertMatte_, "invert", "Invert Matte");

    Bool_knob(f, &linearInput_, "linear_input", "Input is Linear");
    Tooltip(f, "Enable if input is linear (Nuke's default working space).\n"
               "Applies linear → sRGB conversion before inference, as BiRefNet\n"
               "was trained on sRGB images. Disable if your input is already\n"
               "gamma-encoded (sRGB/Rec.709).\n\n"
               "The RGB passthrough and alpha output are unaffected —\n"
               "your linear workflow is preserved.");

    Bool_knob(f, &useImageNetNorm_, "imagenet_norm", "ImageNet Normalize");
    SetFlags(f, Knob::HIDDEN);

    Int_knob(f, &gpuDevice_, "gpu", "GPU Device");
    SetFlags(f, Knob::HIDDEN);

    Divider(f, "");
    Text_knob(f, "BiRefNet_Portrait for Nuke by Peter Mercell, 2026\n"
                 "petermercell.com\n"
                 "\n"
                 "BiRefNet model by Peng Zheng et al.\n"
                 "github.com/ZhengPeng7/BiRefNet (MIT License)\n"
                 "Thank you for the outstanding work and for\n"
                 "open-sourcing BiRefNet!");
}

// ---------------------------------------------------------------------------
// Load TensorRT engine from embedded data (TRT 10.x)
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::loadEngine()
{
    if (engineLoaded_) return;

    const char* engineData = _binary_engine_bin_start;
    size_t engineSize = _binary_engine_bin_end - _binary_engine_bin_start;

    if (engineSize == 0)
    {
        error("Embedded engine data is empty.");
        return;
    }

    std::cerr << "[BiRefNet_Portrait] Loading embedded engine ("
              << (engineSize / (1024 * 1024)) << " MB)..." << std::endl;

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_)
    {
        error("Failed to create TensorRT runtime.");
        return;
    }

    engineTRT_ = runtime_->deserializeCudaEngine(engineData, engineSize);
    if (!engineTRT_)
    {
        error("Failed to deserialize embedded TensorRT engine.");
        return;
    }

    int nbIO = engineTRT_->getNbIOTensors();
    bool foundInput = false, foundOutput = false;

    for (int i = 0; i < nbIO; ++i)
    {
        const char* name = engineTRT_->getIOTensorName(i);
        if (strcmp(name, kInputTensorName) == 0)  foundInput = true;
        if (strcmp(name, kOutputTensorName) == 0) foundOutput = true;
    }

    if (!foundInput)
    {
        error("Engine missing input tensor '%s'", kInputTensorName);
        freeEngine();
        return;
    }
    if (!foundOutput)
    {
        error("Engine missing output tensor '%s'", kOutputTensorName);
        freeEngine();
        return;
    }

    context_ = engineTRT_->createExecutionContext();
    if (!context_)
    {
        error("Failed to create TensorRT execution context.");
        freeEngine();
        return;
    }

    engineLoaded_ = true;
    std::cerr << "[BiRefNet_Portrait] Embedded engine loaded successfully."
              << std::endl;
}

void BiRefNet_Portrait::freeEngine()
{
    if (context_)   { delete context_;   context_   = nullptr; }
    if (engineTRT_) { delete engineTRT_; engineTRT_ = nullptr; }
    if (runtime_)   { delete runtime_;   runtime_   = nullptr; }
    engineLoaded_ = false;
}

// ---------------------------------------------------------------------------
// GPU buffers
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::allocateGPU()
{
    freeGPU();

    size_t inBytes  = 3 * kModelW * kModelH * sizeof(float);
    size_t outBytes = 1 * kModelW * kModelH * sizeof(float);

    cudaSetDevice(gpuDevice_);
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaMalloc(&d_input_,  inBytes));
    CUDA_CHECK(cudaMalloc(&d_output_, outBytes));

    if (!context_->setTensorAddress(kInputTensorName, d_input_))
    {
        error("Failed to set input tensor address");
        return;
    }
    if (!context_->setTensorAddress(kOutputTensorName, d_output_))
    {
        error("Failed to set output tensor address");
        return;
    }
}

void BiRefNet_Portrait::freeGPU()
{
    if (stream_)   { cudaStreamDestroy(stream_);  stream_   = nullptr; }
    if (d_input_)  { cudaFree(d_input_);          d_input_  = nullptr; }
    if (d_output_) { cudaFree(d_output_);         d_output_ = nullptr; }
}

// ---------------------------------------------------------------------------
// _open / _close
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::_open()
{
    loadEngine();
    if (engineLoaded_)
        allocateGPU();
}

void BiRefNet_Portrait::_close()
{
    freeGPU();
    freeEngine();
}

// ---------------------------------------------------------------------------
// _validate
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::_validate(bool for_real)
{
    copy_info();

    ChannelSet out = info().channels();
    out += Chan_Alpha;
    set_out_channels(out);
    info_.turn_on(Chan_Alpha);

    const Format& fmt = info().format();
    frameX_ = fmt.x();
    frameY_ = fmt.y();
    frameW_ = fmt.w();
    frameH_ = fmt.h();

    if (for_real)
    {
        cpuFrameIn_.resize(3 * frameW_ * frameH_, 0.0f);
        cpuMatteOut_.resize(frameW_ * frameH_, 0.0f);
        modelInput_.resize(3 * kModelW * kModelH, 0.0f);
        modelOutput_.resize(kModelW * kModelH, 0.0f);

        inferenceRan_ = false;
    }
}

// ---------------------------------------------------------------------------
// _request
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::_request(int x, int y, int r, int t,
                                 ChannelMask channels, int count)
{
    ChannelSet need = channels;
    need += Chan_Red;
    need += Chan_Green;
    need += Chan_Blue;

    input0().request(frameX_, frameY_,
                     frameX_ + frameW_, frameY_ + frameH_,
                     need, count);
}

// ---------------------------------------------------------------------------
// Fetch all input rows into the planar buffer
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::fetchAllRows()
{
    ChannelSet need(Chan_Red);
    need += Chan_Green;
    need += Chan_Blue;

    float* rPlane = cpuFrameIn_.data();
    float* gPlane = cpuFrameIn_.data() + frameW_ * frameH_;
    float* bPlane = cpuFrameIn_.data() + 2 * frameW_ * frameH_;

    for (int y = frameY_; y < frameY_ + frameH_; ++y)
    {
        Row row(frameX_, frameX_ + frameW_);
        input0().get(y, frameX_, frameX_ + frameW_, need, row);

        if (aborted())
            return;

        const float* rIn = row[Chan_Red]  + frameX_;
        const float* gIn = row[Chan_Green] + frameX_;
        const float* bIn = row[Chan_Blue]  + frameX_;

        int rowIdx = (y - frameY_) * frameW_;

        for (int i = 0; i < frameW_; ++i)
        {
            rPlane[rowIdx + i] = rIn[i];
            gPlane[rowIdx + i] = gIn[i];
            bPlane[rowIdx + i] = bIn[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocess: resize + (optional) linear→sRGB + ImageNet normalize
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::preprocessFrame()
{
    bilinearResize(cpuFrameIn_.data(), frameW_, frameH_,
                   modelInput_.data(), kModelW, kModelH, 3);

    // Linear → sRGB (BiRefNet was trained on gamma-encoded images).
    // Applied on modelInput_ only; cpuFrameIn_ stays linear for RGB passthrough.
    if (linearInput_)
    {
        int n = 3 * kModelW * kModelH;
        float* p = modelInput_.data();
        for (int i = 0; i < n; ++i)
        {
            float v = p[i];
            // Clamp negatives (pow of negative is undefined) but allow >1 (HDR).
            if (v <= 0.0f)
                p[i] = 0.0f;
            else if (v <= 0.0031308f)
                p[i] = 12.92f * v;
            else
                p[i] = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
        }
    }

    if (useImageNetNorm_)
    {
        const float mean[3] = { 0.485f, 0.456f, 0.406f };
        const float std[3]  = { 0.229f, 0.224f, 0.225f };

        for (int c = 0; c < 3; ++c)
        {
            float* ptr = modelInput_.data() + c * kModelW * kModelH;
            int n = kModelW * kModelH;
            for (int i = 0; i < n; ++i)
                ptr[i] = (ptr[i] - mean[c]) / std[c];
        }
    }
}

// ---------------------------------------------------------------------------
// Postprocess: sigmoid + resize back
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::postprocessMatte()
{
    int n = kModelW * kModelH;

    for (int i = 0; i < n; ++i)
    {
        float v = modelOutput_[i];
        modelOutput_[i] = 1.0f / (1.0f + std::exp(-v));
    }

    bilinearResize(modelOutput_.data(), kModelW, kModelH,
                   cpuMatteOut_.data(), frameW_, frameH_, 1);

    if (invertMatte_)
    {
        int pixels = frameW_ * frameH_;
        for (int i = 0; i < pixels; ++i)
            cpuMatteOut_[i] = 1.0f - cpuMatteOut_[i];
    }
}

// ---------------------------------------------------------------------------
// TensorRT inference (TRT 10.x: enqueueV3)
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::runInference()
{
    if (!engineLoaded_ || !context_) return;

    size_t inBytes  = 3 * kModelW * kModelH * sizeof(float);
    size_t outBytes = 1 * kModelW * kModelH * sizeof(float);

    cudaMemcpyAsync(d_input_, modelInput_.data(), inBytes,
                    cudaMemcpyHostToDevice, stream_);

    bool ok = context_->enqueueV3(stream_);
    if (!ok)
        std::cerr << "[BiRefNet_Portrait] enqueueV3 FAILED!" << std::endl;

    cudaMemcpyAsync(modelOutput_.data(), d_output_, outBytes,
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
}

// ---------------------------------------------------------------------------
// Full inference pipeline
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::doFullInference()
{
    fetchAllRows();

    if (aborted()) return;

    preprocessFrame();
    runInference();
    postprocessMatte();
}

// ---------------------------------------------------------------------------
// engine() — called per scanline by Nuke (multi-threaded)
// ---------------------------------------------------------------------------
void BiRefNet_Portrait::engine(int y, int x, int r,
                               ChannelMask channels, Row& row)
{
    if (!engineLoaded_)
    {
        input0().get(y, x, r, channels, row);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(inferenceMutex_);
        if (!inferenceRan_)
        {
            doFullInference();
            inferenceRan_ = true;
        }
    }

    int localY = y - frameY_;
    int rowOffset = localY * frameW_;

    float* rOut = row.writable(Chan_Red);
    float* gOut = row.writable(Chan_Green);
    float* bOut = row.writable(Chan_Blue);

    if (outputMatteOnly_)
    {
        for (int i = x; i < r; ++i)
        {
            int localX = i - frameX_;
            float m = cpuMatteOut_[rowOffset + localX];
            rOut[i] = m;
            gOut[i] = m;
            bOut[i] = m;
        }
    }
    else
    {
        const float* rPlane = cpuFrameIn_.data();
        const float* gPlane = cpuFrameIn_.data() + frameW_ * frameH_;
        const float* bPlane = cpuFrameIn_.data() + 2 * frameW_ * frameH_;

        for (int i = x; i < r; ++i)
        {
            int localX = i - frameX_;
            rOut[i] = rPlane[rowOffset + localX];
            gOut[i] = gPlane[rowOffset + localX];
            bOut[i] = bPlane[rowOffset + localX];
        }
    }

    if (channels & Mask_Alpha)
    {
        float* aOut = row.writable(Chan_Alpha);
        for (int i = x; i < r; ++i)
        {
            int localX = i - frameX_;
            aOut[i] = cpuMatteOut_[rowOffset + localX];
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
static Iop* build(Node* node)
{
    return new BiRefNet_Portrait(node);
}

const Iop::Description BiRefNet_Portrait::description(
    "BiRefNet_Portrait",
    "AI/BiRefNet_Portrait",
    build
);
