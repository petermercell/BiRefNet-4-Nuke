/*
 * TRTBiRefNet - TensorRT BiRefNet Inference Node for Nuke
 *
 * Model details (confirmed via Netron / trtexec):
 *   Input:  "input_image"  float32[1,3,1024,1024]  (NCHW)
 *   Output: "output_image" float32[1,1,1024,1024]  (NCHW)
 *   Last op: Conv (decoder.conv_out1.conv_out1.0.Conv) — raw logits
 *   Sigmoid applied by this node.
 *
 * Built for: TensorRT 10.15.1 (enqueueV3 + named tensors)
 *            Nuke 17.0 NDK
 *            CUDA 13.1
 *
 * Author: Peter Mercell
 * Website: petermercell.com
 */

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Thread.h"
#include "DDImage/Format.h"
#include "DDImage/Interest.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <iostream>
#include <mutex>

using namespace DD::Image;

// ---------------------------------------------------------------------------
// TensorRT logger
// ---------------------------------------------------------------------------
class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRTBiRefNet] " << msg << std::endl;
    }
};

static TRTLogger gLogger;

// ---------------------------------------------------------------------------
// Tensor names (confirmed from Netron / trtexec)
// ---------------------------------------------------------------------------
static const char* kInputTensorName  = "input_image";
static const char* kOutputTensorName = "output_image";

// ---------------------------------------------------------------------------
// Resolution presets
// ---------------------------------------------------------------------------
static const char* const kResolutionNames[] = {
    "512x512",
    "768x768",
    "1024x1024",
    "2048x2048",
    "Custom",
    nullptr
};

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
// TRTBiRefNet node
// ---------------------------------------------------------------------------
class TRTBiRefNet : public Iop
{

public:

    TRTBiRefNet(Node* node);
    ~TRTBiRefNet() override;

    const char* Class() const override { return description.name; }
    const char* node_help() const override
    {
        return "TensorRT BiRefNet inference.\n\n"
               "Runs a TensorRT engine of BiRefNet to produce\n"
               "a foreground matte in the alpha channel.\n\n"
               "Input: RGB image (any resolution)\n"
               "Output: RGBA (alpha = BiRefNet matte)\n\n"
               "Engine tensors:\n"
               "  input:  input_image  [1,3,1024,1024]\n"
               "  output: output_image [1,1,1024,1024]\n\n"
               "Output is raw logits — sigmoid applied by this node.\n\n"
               "TRTBiRefNet by Peter Mercell, 2026\n"
               "petermercell.com\n\n"
               "BiRefNet model by Peng Zheng et al.\n"
               "https://github.com/ZhengPeng7/BiRefNet";
    }

    void knobs(Knob_Callback f) override;
    int knob_changed(Knob* k) override;

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
    const char* enginePath_;
    int         resolutionEnum_;
    int         modelW_;
    int         modelH_;
    bool        outputMatteOnly_;
    bool        invertMatte_;
    bool        useImageNetNorm_;
    int         gpuDevice_;

    // --- frame geometry ---
    int frameW_;
    int frameH_;
    int frameX_;  // format origin
    int frameY_;

    // --- CPU buffers ---
    std::vector<float> cpuFrameIn_;   // 3 * frameW * frameH (planar CHW)
    std::vector<float> cpuMatteOut_;  // frameW * frameH
    std::vector<float> modelInput_;   // 3 * modelW * modelH
    std::vector<float> modelOutput_;  // modelW * modelH

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

    int resolvedModelW() const;
    int resolvedModelH() const;
};

// ---------------------------------------------------------------------------
// Ctor / Dtor
// ---------------------------------------------------------------------------
TRTBiRefNet::TRTBiRefNet(Node* node)
    : Iop(node)
    , enginePath_("")
    , resolutionEnum_(2)
    , modelW_(1024)
    , modelH_(1024)
    , outputMatteOnly_(false)
    , invertMatte_(false)
    , useImageNetNorm_(true)
    , gpuDevice_(0)
    , frameW_(0), frameH_(0), frameX_(0), frameY_(0)
    , d_input_(nullptr), d_output_(nullptr), stream_(nullptr)
    , runtime_(nullptr), engineTRT_(nullptr), context_(nullptr)
    , inferenceRan_(false)
    , engineLoaded_(false)
{
}

TRTBiRefNet::~TRTBiRefNet()
{
    freeGPU();
    freeEngine();
}

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------
void TRTBiRefNet::knobs(Knob_Callback f)
{
    File_knob(f, &enginePath_, "engine_path", "Engine File");
    Tooltip(f, "Path to the TensorRT .engine file.\n"
               "e.g. BiRefNet-portrait-epoch_150_fp16.engine");

    Divider(f, "Model");

    Enumeration_knob(f, &resolutionEnum_, kResolutionNames,
                     "resolution", "Resolution");
    Tooltip(f, "Model input resolution. Must match the engine.");

    Int_knob(f, &modelW_, "model_w", "Custom W");
    Int_knob(f, &modelH_, "model_h", "Custom H");
    SetFlags(f, Knob::HIDDEN);

    Divider(f, "Output");

    Bool_knob(f, &outputMatteOnly_, "matte_only", "Matte Only (BW)");
    Tooltip(f, "Output matte as greyscale RGB.");

    Bool_knob(f, &invertMatte_, "invert", "Invert Matte");

    Divider(f, "Advanced");

    Bool_knob(f, &useImageNetNorm_, "imagenet_norm", "ImageNet Normalize");
    Tooltip(f, "Apply ImageNet mean/std normalization.\n"
               "Required for standard BiRefNet weights.");

    Int_knob(f, &gpuDevice_, "gpu", "GPU Device");

    Divider(f, "");
    Text_knob(f, "TRTBiRefNet by Peter Mercell, 2026\n"
                 "petermercell.com\n"
                 "BiRefNet model: github.com/ZhengPeng7/BiRefNet");
}

int TRTBiRefNet::knob_changed(Knob* k)
{
    if (k->is("resolution"))
    {
        bool custom = (resolutionEnum_ == 4);
        knob("model_w")->visible(custom);
        knob("model_h")->visible(custom);
        return 1;
    }
    return Iop::knob_changed(k);
}

// ---------------------------------------------------------------------------
// Resolution helpers
// ---------------------------------------------------------------------------
int TRTBiRefNet::resolvedModelW() const
{
    switch (resolutionEnum_) {
        case 0: return 512;
        case 1: return 768;
        case 2: return 1024;
        case 3: return 2048;
        default: return modelW_;
    }
}

int TRTBiRefNet::resolvedModelH() const
{
    switch (resolutionEnum_) {
        case 0: return 512;
        case 1: return 768;
        case 2: return 1024;
        case 3: return 2048;
        default: return modelH_;
    }
}

// ---------------------------------------------------------------------------
// Load TensorRT engine (TRT 10.x)
// ---------------------------------------------------------------------------
void TRTBiRefNet::loadEngine()
{
    if (engineLoaded_) return;
    if (!enginePath_ || strlen(enginePath_) == 0)
    {
        error("No TensorRT engine file specified.");
        return;
    }

    std::ifstream file(enginePath_, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        error("Cannot open engine file: %s", enginePath_);
        return;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize))
    {
        error("Failed to read engine file.");
        return;
    }
    file.close();

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_)
    {
        error("Failed to create TensorRT runtime.");
        return;
    }

    engineTRT_ = runtime_->deserializeCudaEngine(buffer.data(), fileSize);
    if (!engineTRT_)
    {
        error("Failed to deserialize TensorRT engine from: %s", enginePath_);
        return;
    }

    // Verify tensor names (TRT 10.x API)
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
    std::cerr << "[TRTBiRefNet] Engine loaded: " << enginePath_ << std::endl;
}

void TRTBiRefNet::freeEngine()
{
    if (context_)   { delete context_;   context_   = nullptr; }
    if (engineTRT_) { delete engineTRT_; engineTRT_ = nullptr; }
    if (runtime_)   { delete runtime_;   runtime_   = nullptr; }
    engineLoaded_ = false;
}

// ---------------------------------------------------------------------------
// GPU buffers
// ---------------------------------------------------------------------------
void TRTBiRefNet::allocateGPU()
{
    freeGPU();

    int mW = resolvedModelW();
    int mH = resolvedModelH();

    size_t inBytes  = 3 * mW * mH * sizeof(float);
    size_t outBytes = 1 * mW * mH * sizeof(float);

    cudaSetDevice(gpuDevice_);
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaMalloc(&d_input_,  inBytes));
    CUDA_CHECK(cudaMalloc(&d_output_, outBytes));

    // TRT 10.x: set tensor addresses by name
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

void TRTBiRefNet::freeGPU()
{
    if (stream_)   { cudaStreamDestroy(stream_);  stream_   = nullptr; }
    if (d_input_)  { cudaFree(d_input_);          d_input_  = nullptr; }
    if (d_output_) { cudaFree(d_output_);         d_output_ = nullptr; }
}

// ---------------------------------------------------------------------------
// _open / _close
// ---------------------------------------------------------------------------
void TRTBiRefNet::_open()
{
    loadEngine();
    if (engineLoaded_)
        allocateGPU();
}

void TRTBiRefNet::_close()
{
    freeGPU();
    freeEngine();
}

// ---------------------------------------------------------------------------
// _validate
// ---------------------------------------------------------------------------
void TRTBiRefNet::_validate(bool for_real)
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
        int mW = resolvedModelW();
        int mH = resolvedModelH();

        cpuFrameIn_.resize(3 * frameW_ * frameH_, 0.0f);
        cpuMatteOut_.resize(frameW_ * frameH_, 0.0f);
        modelInput_.resize(3 * mW * mH, 0.0f);
        modelOutput_.resize(mW * mH, 0.0f);

        inferenceRan_ = false;
    }
}

// ---------------------------------------------------------------------------
// _request — request full frame from input
// ---------------------------------------------------------------------------
void TRTBiRefNet::_request(int x, int y, int r, int t,
                           ChannelMask channels, int count)
{
    ChannelSet need = channels;
    need += Chan_Red;
    need += Chan_Green;
    need += Chan_Blue;

    // Request the full format area
    input0().request(frameX_, frameY_,
                     frameX_ + frameW_, frameY_ + frameH_,
                     need, count);
}

// ---------------------------------------------------------------------------
// Fetch all input rows into the planar buffer
// Called once under lock — no threading issue
// ---------------------------------------------------------------------------
void TRTBiRefNet::fetchAllRows()
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
// Preprocess: resize + ImageNet normalize
// ---------------------------------------------------------------------------
void TRTBiRefNet::preprocessFrame()
{
    int mW = resolvedModelW();
    int mH = resolvedModelH();

    bilinearResize(cpuFrameIn_.data(), frameW_, frameH_,
                   modelInput_.data(), mW, mH, 3);

    if (useImageNetNorm_)
    {
        const float mean[3] = { 0.485f, 0.456f, 0.406f };
        const float std[3]  = { 0.229f, 0.224f, 0.225f };

        for (int c = 0; c < 3; ++c)
        {
            float* ptr = modelInput_.data() + c * mW * mH;
            int n = mW * mH;
            for (int i = 0; i < n; ++i)
                ptr[i] = (ptr[i] - mean[c]) / std[c];
        }
    }
}

// ---------------------------------------------------------------------------
// Postprocess: sigmoid + resize back
// Last ONNX op is Conv — output is raw logits, sigmoid needed
// ---------------------------------------------------------------------------
void TRTBiRefNet::postprocessMatte()
{
    int mW = resolvedModelW();
    int mH = resolvedModelH();
    int n  = mW * mH;

    for (int i = 0; i < n; ++i)
    {
        float v = modelOutput_[i];
        modelOutput_[i] = 1.0f / (1.0f + std::exp(-v));
    }

    bilinearResize(modelOutput_.data(), mW, mH,
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
void TRTBiRefNet::runInference()
{
    if (!engineLoaded_ || !context_) return;

    int mW = resolvedModelW();
    int mH = resolvedModelH();

    size_t inBytes  = 3 * mW * mH * sizeof(float);
    size_t outBytes = 1 * mW * mH * sizeof(float);

    cudaMemcpyAsync(d_input_, modelInput_.data(), inBytes,
                    cudaMemcpyHostToDevice, stream_);

    bool ok = context_->enqueueV3(stream_);
    if (!ok)
        std::cerr << "[TRTBiRefNet] enqueueV3 FAILED!" << std::endl;

    cudaMemcpyAsync(modelOutput_.data(), d_output_, outBytes,
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
}

// ---------------------------------------------------------------------------
// Full inference pipeline (called once per frame under lock)
// ---------------------------------------------------------------------------
void TRTBiRefNet::doFullInference()
{
    fetchAllRows();

    if (aborted()) return;

    preprocessFrame();
    runInference();
    postprocessMatte();
}

// ---------------------------------------------------------------------------
// engine() — called per scanline by Nuke (multi-threaded)
//
// SOLUTION TO DEADLOCK:
//   The first thread to enter acquires the lock, fetches ALL input rows
//   from the upstream node, runs the full inference pipeline, and sets
//   inferenceRan_ = true.
//   All subsequent threads see inferenceRan_ == true and just read
//   their row from the pre-computed output buffer.
//   No condition variable needed — no deadlock possible.
// ---------------------------------------------------------------------------
void TRTBiRefNet::engine(int y, int x, int r,
                         ChannelMask channels, Row& row)
{
    if (!engineLoaded_)
    {
        input0().get(y, x, r, channels, row);
        return;
    }

    // First thread does the full fetch + inference
    {
        std::lock_guard<std::mutex> lock(inferenceMutex_);
        if (!inferenceRan_)
        {
            doFullInference();
            inferenceRan_ = true;
        }
    }

    // Now read from pre-computed buffers
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
        // Pass through input RGB from our stored buffer
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
    return new TRTBiRefNet(node);
}

const Iop::Description TRTBiRefNet::description(
    "TRTBiRefNet",
    "AI/TRTBiRefNet",
    build
);
