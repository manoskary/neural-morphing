#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <juce_audio_basics/juce_audio_basics.h>

struct TokenBlock
{
    int batchSize = 1;
    int codebooks = 0;
    int frames = 0;
    std::vector<int32_t> tokens;

    bool empty() const noexcept { return tokens.empty() || frames == 0 || codebooks == 0; }

    int index(int codebook, int frame) const noexcept
    {
        return codebook * frames + frame;
    }
};

enum class TokenLayout
{
    CodebookMajor = 0,
    FrameMajor
};

class ModelBackend
{
public:
    virtual ~ModelBackend() = default;

    virtual bool load(const std::string& modelRoot) = 0;
    virtual bool ready() const = 0;
    virtual int sampleRate() const = 0;
    virtual int codebookCount() const = 0;
    virtual int embeddingDimension() const = 0;
    virtual int requiredInputChannels() const { return 1; }
    virtual double frameRateHz() const { return 0.0; }
    virtual TokenLayout tokenLayout() const { return TokenLayout::CodebookMajor; }
    virtual bool setCodec(const std::string& codecId)
    {
        (void) codecId;
        return false;
    }

    virtual TokenBlock encodePCM(const juce::AudioBuffer<float>& mono) = 0;
    virtual std::vector<float> tokensToVectorRow(const TokenBlock& block, int frameIndex) = 0;
    virtual juce::AudioBuffer<float> decodeTokens(const TokenBlock& block) = 0;
};

std::unique_ptr<ModelBackend> createStubModelBackend();
