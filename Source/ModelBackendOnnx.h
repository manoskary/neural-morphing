#pragma once

#include <memory>

#include <juce_core/juce_core.h>

#include "ModelBackend.h"

class ModelBackendOnnx : public ModelBackend
{
public:
    ModelBackendOnnx();
    ~ModelBackendOnnx() override;

    bool load(const std::string& modelRoot) override;
    bool ready() const override { return ready_; }
    int sampleRate() const override { return sampleRate_; }
    int codebookCount() const override { return numCodebooks_; }
    int embeddingDimension() const override { return embeddingDim_; }
    int requiredInputChannels() const override { return 1; }
    double frameRateHz() const override { return 0.0; }
    TokenLayout tokenLayout() const override { return TokenLayout::CodebookMajor; }

    TokenBlock encodePCM(const juce::AudioBuffer<float>& mono) override;
    std::vector<float> tokensToVectorRow(const TokenBlock& block, int frameIndex) override;
    juce::AudioBuffer<float> decodeTokens(const TokenBlock& block) override;

private:
    bool ready_ = false;
    int sampleRate_ = 44100;
    int numCodebooks_ = 0;
    int codebookSize_ = 0;
    int embeddingDimPerCodebook_ = 0;
    int embeddingDim_ = 0;

    std::vector<float> embeddings_;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<ModelBackend> createOnnxModelBackend();
