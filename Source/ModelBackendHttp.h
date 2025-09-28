#pragma once

#if NM_WITH_PYBRIDGE

#include <atomic>
#include <memory>

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

    TokenBlock encodePCM(const juce::AudioBuffer<float>& mono) override;
    std::vector<float> tokensToVectorRow(const TokenBlock& block, int frameIndex) override;
    juce::AudioBuffer<float> decodeTokens(const TokenBlock& block) override;

    // HTTP backend specific methods
    void setServerUrl(const juce::String& url);
    juce::String getServerUrl() const;
    bool checkHealth();
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

    juce::String serverUrl_;
    std::atomic<bool> ready_{ false };
    int sampleRate_ = 44100;
    int codebookCount_ = 0;
    int embeddingDim_ = 0;
    mutable juce::String lastError_;
    mutable juce::CriticalSection errorMutex_;
    
    static constexpr int timeoutMs_ = 10000; // 10 second timeout
};

std::unique_ptr<ModelBackend> createHttpModelBackend(const juce::String& serverUrl = "http://localhost:8000");

#endif // NM_WITH_PYBRIDGE