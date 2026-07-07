#include "ModelBackend.h"

#include <cmath>
#include <random>

namespace
{
class StubModelBackend : public ModelBackend
{
public:
    bool load(const std::string& /*modelRoot*/) override
    {
        ready_ = true;
        return true;
    }

    bool ready() const override
    {
        return ready_;
    }

    int sampleRate() const override
    {
        return sampleRate_;
    }

    int codebookCount() const override
    {
        return 1;
    }

    int embeddingDimension() const override
    {
        return 2;
    }

    int requiredInputChannels() const override
    {
        return 1;
    }

    double frameRateHz() const override
    {
        return static_cast<double>(sampleRate_) / static_cast<double>(frameSize_);
    }

    TokenLayout tokenLayout() const override
    {
        return TokenLayout::CodebookMajor;
    }

    TokenBlock encodePCM(const juce::AudioBuffer<float>& mono) override
    {
        const int samples = mono.getNumSamples();
        TokenBlock block;
        block.batchSize = 1;
        block.codebooks = 1;
        block.frames = juce::jmax(1, samples / frameSize_);
        block.tokens.resize(block.codebooks * block.frames, 0);

        const float* data = mono.getReadPointer(0);
        for (int i = 0; i < block.frames; ++i)
        {
            const int start = i * frameSize_;
            const int end = juce::jmin(start + frameSize_, samples);
            float energy = 0.0f;
            for (int j = start; j < end; ++j)
                energy += std::abs(data[j]);
            energy = (end > start) ? energy / static_cast<float>(end - start) : 0.0f;
            block.tokens[i] = static_cast<int32_t>(juce::jlimit(0, tokenRange_ - 1, static_cast<int>(energy * tokenRange_)));
        }

        return block;
    }

    std::vector<float> tokensToVectorRow(const TokenBlock& block, int frameIndex) override
    {
        if (frameIndex < 0 || frameIndex >= block.frames)
            return {};

        const int token = block.tokens[block.index(0, frameIndex)];
        return { static_cast<float>(token) / static_cast<float>(tokenRange_), 1.0f - static_cast<float>(token) / static_cast<float>(tokenRange_) };
    }

    juce::AudioBuffer<float> decodeTokens(const TokenBlock& block) override
    {
        juce::AudioBuffer<float> buffer(2, block.frames * frameSize_);
        buffer.clear();

        std::mt19937 rng(block.frames);
        std::uniform_real_distribution<float> noise(-0.05f, 0.05f);

        for (int frame = 0; frame < block.frames; ++frame)
        {
            const int token = (frame < static_cast<int>(block.tokens.size())) ? block.tokens[frame] : 0;
            const float level = static_cast<float>(token) / static_cast<float>(juce::jmax(1, tokenRange_));
            const float freq = 220.0f + level * 880.0f;

            for (int sample = 0; sample < frameSize_; ++sample)
            {
                const int index = frame * frameSize_ + sample;
                const float t = static_cast<float>(index) / static_cast<float>(sampleRate_);
                const float value = std::sin(juce::MathConstants<float>::twoPi * freq * t) * (0.2f + level * 0.8f) + noise(rng);
                buffer.setSample(0, index, value);
                buffer.setSample(1, index, value);
            }
        }

        return buffer;
    }

private:
    static constexpr int sampleRate_ = 44100;
    static constexpr int frameSize_ = 256;
    static constexpr int tokenRange_ = 128;

    bool ready_ = false;
};
}

std::unique_ptr<ModelBackend> createStubModelBackend()
{
    return std::make_unique<StubModelBackend>();
}
