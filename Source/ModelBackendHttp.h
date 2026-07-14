#pragma once

#if NM_WITH_PYBRIDGE

#include <atomic>
#include <memory>
#include <vector>

#include <juce_core/juce_core.h>

#include "ModelBackend.h"

class ModelBackendHttp : public ModelBackend
{
public:
    explicit ModelBackendHttp(const juce::String& serverUrl = "http://localhost:8000");
    ~ModelBackendHttp() override;

    bool load(const std::string& modelRoot) override;
    bool ready() const override { return ready_.load(); }
    int sampleRate() const override { return sampleRate_; }
    int codebookCount() const override { return codebookCount_; }
    int embeddingDimension() const override { return embeddingDim_; }
    int requiredInputChannels() const override { return requiredInputChannels_; }
    double frameRateHz() const override { return frameRateHz_; }
    TokenLayout tokenLayout() const override { return tokenLayout_; }
    bool setCodec(const std::string& codecId) override;

    TokenBlock encodePCM(const juce::AudioBuffer<float>& mono, double sourceSampleRate) override;
    std::vector<float> tokensToVectorRow(const TokenBlock& block, int frameIndex) override;
    bool tokensToVectorRows(const TokenBlock& block,
                            int startFrame,
                            int frameCount,
                            std::vector<std::vector<float>>& out) override;
    juce::AudioBuffer<float> decodeTokens(const TokenBlock& block) override;

    // HTTP backend specific methods
    void setServerUrl(const juce::String& url);
    juce::String getServerUrl() const;
    bool checkHealth();
    bool refreshCapabilities();
    juce::String activeCodec() const;
    juce::String getLastError() const;

private:
    struct HttpResponse
    {
        bool success = false;
        int statusCode = 0;
        juce::String body;
        juce::String error;
    };

    HttpResponse makeHttpRequest(const juce::String& endpoint, const juce::String& method = "GET", 
                                const juce::String& jsonBody = juce::String()) const;
    juce::String encodeBase64(const juce::MemoryBlock& data) const;
    juce::MemoryBlock decodeBase64(const juce::String& encoded) const;
    bool parseCapabilities(const juce::String& responseBody);
    TokenBlock parseTokenBlockResponse(const juce::String& responseBody, bool* ok = nullptr) const;

    juce::String serverUrl_;
    std::atomic<bool> ready_{ false };
    int sampleRate_ = 44100;
    int codebookCount_ = 0;
    int embeddingDim_ = 0;
    int requiredInputChannels_ = 1;
    double frameRateHz_ = 0.0;
    TokenLayout tokenLayout_ = TokenLayout::CodebookMajor;
    bool supportsPcmEndpoints_ = false;
    juce::String activeCodec_ = "dac";
    std::vector<juce::String> supportedCodecs_;
    mutable juce::String lastError_;
    mutable juce::CriticalSection errorMutex_;
    
    int timeoutMs_ = 30000; // default timeout, overridable via env
};

std::unique_ptr<ModelBackend> createHttpModelBackend(const juce::String& serverUrl = "http://localhost:8000");

#endif // NM_WITH_PYBRIDGE
