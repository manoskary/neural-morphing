#include "ModelBackendOnnx.h"

#include <algorithm>
#include <array>
#include <memory>

#include <onnxruntime_cxx_api.h>

#include <juce_data_structures/juce_data_structures.h>

#include "ModelBackend.h"

namespace
{
#if defined(_WIN32)
std::wstring toOrtPath(const juce::File& file)
{
    return std::wstring(file.getFullPathName().toWideCharPointer());
}
#else
std::string toOrtPath(const juce::File& file)
{
    return file.getFullPathName().toStdString();
}
#endif

Ort::Env& getOrtEnv()
{
    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "NeuralMorphing"};
    return env;
}

std::unique_ptr<Ort::Session> createSession(const juce::File& file, Ort::SessionOptions& options)
{
    if (!file.existsAsFile())
        return nullptr;

    auto path = toOrtPath(file);
    return std::make_unique<Ort::Session>(getOrtEnv(), path.c_str(), options);
}

juce::var parseMetadata(const juce::File& file)
{
    auto stream = file.createInputStream();
    if (stream == nullptr)
        return {};

    auto jsonText = stream->readEntireStreamAsString();
    if (jsonText.isEmpty())
        return {};

    return juce::JSON::parse(jsonText);
}

bool readEmbeddingBinary(const juce::File& file, std::vector<float>& data)
{
    auto stream = file.createInputStream();
    if (stream == nullptr)
        return false;

    const auto expectedBytes = static_cast<int>(sizeof(float) * data.size());
    const auto bytesRead = stream->read(data.data(), expectedBytes);
    return bytesRead == expectedBytes;
}

Ort::MemoryInfo& cpuMemory()
{
    static Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    return memInfo;
}

} // namespace

class ModelBackendOnnx::Impl
{
public:
    Impl()
    {
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }

    bool load(const juce::File& root)
    {
        auto encoderFile = root.getChildFile("encoder.onnx");
        auto decoderFile = root.getChildFile("decoder.onnx");

        encoder = createSession(encoderFile, sessionOptions);
        decoder = createSession(decoderFile, sessionOptions);
        return encoder != nullptr && decoder != nullptr;
    }

    std::unique_ptr<Ort::Session> encoder;
    std::unique_ptr<Ort::Session> decoder;
    Ort::SessionOptions sessionOptions;
};

ModelBackendOnnx::ModelBackendOnnx() = default;
ModelBackendOnnx::~ModelBackendOnnx() = default;

bool ModelBackendOnnx::load(const std::string& modelRoot)
{
    ready_ = false;
    juce::File root(modelRoot);
    if (!root.isDirectory())
        return false;

    auto metadataVar = parseMetadata(root.getChildFile("metadata.json"));
    if (!metadataVar.isObject())
        return false;

    const auto* obj = metadataVar.getDynamicObject();
    if (obj == nullptr)
        return false;

    sampleRate_ = static_cast<int>(obj->getProperty("sample_rate"));
    numCodebooks_ = static_cast<int>(obj->getProperty("num_codebooks"));
    codebookSize_ = static_cast<int>(obj->getProperty("codebook_size"));
    embeddingDimPerCodebook_ = static_cast<int>(obj->getProperty("embedding_dim"));
    embeddingDim_ = embeddingDimPerCodebook_ * numCodebooks_;
    auto frameRateVar = obj->getProperty("frame_rate_hz");
    frameRateHz_ = frameRateVar.isVoid() ? 0.0 : static_cast<double>(frameRateVar);

    auto embeddingFileVar = obj->getProperty("embedding_file");
    const juce::String embeddingFileName = embeddingFileVar.isVoid() ? "embeddings.bin" : embeddingFileVar.toString();
    const juce::File embeddingFile = root.getChildFile(embeddingFileName);

    const size_t totalValues = static_cast<size_t>(numCodebooks_) * static_cast<size_t>(codebookSize_) * static_cast<size_t>(embeddingDimPerCodebook_);
    embeddings_.resize(totalValues);
    if (!readEmbeddingBinary(embeddingFile, embeddings_))
        return false;

    impl_ = std::make_unique<Impl>();
    if (!impl_->load(root))
    {
        impl_.reset();
        return false;
    }

    ready_ = true;
    return true;
}

TokenBlock ModelBackendOnnx::encodePCM(const juce::AudioBuffer<float>& mono)
{
    if (!ready_ || mono.getNumSamples() == 0 || impl_ == nullptr)
        return {};

    const int samples = mono.getNumSamples();
    std::vector<float> input(static_cast<size_t>(samples));
    const float* src = mono.getReadPointer(0);
    std::copy(src, src + samples, input.begin());

    std::array<int64_t, 2> inputShape{1, static_cast<int64_t>(samples)};

    auto inputTensor = Ort::Value::CreateTensor<float>(cpuMemory(), input.data(), input.size(), inputShape.data(), inputShape.size());
    const char* inputName = "waveform";
    const char* outputName = "tokens";
    Ort::RunOptions runOptions(nullptr);
    auto outputs = impl_->encoder->Run(runOptions, &inputName, &inputTensor, 1, &outputName, 1);

    if (outputs.empty())
        return {};

    auto& outTensor = outputs.front();
    auto typeInfo = outTensor.GetTensorTypeAndShapeInfo();
    auto dims = typeInfo.GetShape();
    if (dims.size() < 2)
        return {};

    const int codebooks = static_cast<int>(dims[dims.size() - 2]);
    const int frames = static_cast<int>(dims.back());

    const int64_t* rawTokens = outTensor.GetTensorData<int64_t>();
    if (rawTokens == nullptr)
        return {};

    TokenBlock block;
    block.batchSize = 1;
    block.codebooks = codebooks;
    block.frames = frames;
    block.tokens.resize(static_cast<size_t>(codebooks * frames));

    for (int cb = 0; cb < codebooks; ++cb)
    {
        for (int frame = 0; frame < frames; ++frame)
        {
            const size_t idx = static_cast<size_t>(cb) * static_cast<size_t>(frames) + static_cast<size_t>(frame);
            block.tokens[idx] = static_cast<int32_t>(rawTokens[idx]);
        }
    }

    return block;
}

std::vector<float> ModelBackendOnnx::tokensToVectorRow(const TokenBlock& block, int frameIndex)
{
    std::vector<float> row;
    if (!ready_ || block.empty() || frameIndex < 0 || frameIndex >= block.frames)
        return row;

    row.resize(static_cast<size_t>(embeddingDim_));

    const size_t codebookStride = static_cast<size_t>(codebookSize_) * static_cast<size_t>(embeddingDimPerCodebook_);

    for (int cb = 0; cb < block.codebooks; ++cb)
    {
        const int token = block.tokens[block.index(cb, frameIndex)];
        const int safeToken = juce::jlimit(0, codebookSize_ - 1, token);
        const size_t embeddingOffset = static_cast<size_t>(cb) * codebookStride + static_cast<size_t>(safeToken) * static_cast<size_t>(embeddingDimPerCodebook_);
        const size_t rowOffset = static_cast<size_t>(cb) * static_cast<size_t>(embeddingDimPerCodebook_);
        std::copy_n(embeddings_.data() + embeddingOffset, static_cast<size_t>(embeddingDimPerCodebook_), row.begin() + rowOffset);
    }

    return row;
}

juce::AudioBuffer<float> ModelBackendOnnx::decodeTokens(const TokenBlock& block)
{
    juce::AudioBuffer<float> buffer;
    if (!ready_ || block.empty() || impl_ == nullptr)
        return buffer;

    const int frames = block.frames;
    const int codebooks = block.codebooks;
    std::vector<int64_t> tokens64(static_cast<size_t>(codebooks * frames));

    for (int cb = 0; cb < codebooks; ++cb)
        for (int frame = 0; frame < frames; ++frame)
            tokens64[static_cast<size_t>(cb) * static_cast<size_t>(frames) + static_cast<size_t>(frame)] = block.tokens[block.index(cb, frame)];

    std::array<int64_t, 2> inputShape{static_cast<int64_t>(codebooks), static_cast<int64_t>(frames)};
    auto inputTensor = Ort::Value::CreateTensor<int64_t>(cpuMemory(), tokens64.data(), tokens64.size(), inputShape.data(), inputShape.size());

    const char* inputName = "tokens";
    const char* outputName = "audio";
    Ort::RunOptions runOptions(nullptr);
    auto outputs = impl_->decoder->Run(runOptions, &inputName, &inputTensor, 1, &outputName, 1);
    if (outputs.empty())
        return buffer;

    auto& outTensor = outputs.front();
    auto typeInfo = outTensor.GetTensorTypeAndShapeInfo();
    auto dims = typeInfo.GetShape();
    if (dims.empty())
        return buffer;

    const int channels = dims.size() == 1 ? 1 : static_cast<int>(dims[dims.size() - 2]);
    const int samples = static_cast<int>(dims.back());

    const float* audioData = outTensor.GetTensorData<float>();
    if (audioData == nullptr)
        return buffer;

    buffer.setSize(channels, samples, false, false, true);

    for (int ch = 0; ch < channels; ++ch)
    {
        float* dst = buffer.getWritePointer(ch);
        for (int sample = 0; sample < samples; ++sample)
        {
            const size_t idx = static_cast<size_t>(ch) * static_cast<size_t>(samples) + static_cast<size_t>(sample);
            dst[sample] = audioData[idx];
        }
    }

    return buffer;
}

std::unique_ptr<ModelBackend> createOnnxModelBackend()
{
    return std::make_unique<ModelBackendOnnx>();
}
