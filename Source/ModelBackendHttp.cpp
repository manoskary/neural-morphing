#include "ModelBackendHttp.h"

#if NM_WITH_PYBRIDGE

#include "JuceHeader.h"

#include <algorithm>
#include <cstring>

namespace
{
TokenLayout tokenLayoutFromString(const juce::String& layout)
{
    if (layout.equalsIgnoreCase("frame_major"))
        return TokenLayout::FrameMajor;

    return TokenLayout::CodebookMajor;
}

juce::String tokenLayoutToString(TokenLayout layout)
{
    if (layout == TokenLayout::FrameMajor)
        return "frame_major";

    return "codebook_major";
}

juce::var getPropertyOrDefault(const juce::DynamicObject* obj, const juce::Identifier& property, const juce::var& fallback)
{
    if (obj == nullptr)
        return fallback;

    auto value = obj->getProperty(property);
    return value.isVoid() ? fallback : value;
}

std::vector<int32_t> frameMajorToCodebookMajor(const std::vector<int32_t>& frameMajor, int codebooks, int frames)
{
    std::vector<int32_t> result;
    if (codebooks <= 0 || frames <= 0)
        return result;

    result.resize(static_cast<size_t>(codebooks * frames), 0);
    for (int frame = 0; frame < frames; ++frame)
    {
        for (int codebook = 0; codebook < codebooks; ++codebook)
        {
            const size_t src = static_cast<size_t>(frame) * static_cast<size_t>(codebooks) + static_cast<size_t>(codebook);
            const size_t dst = static_cast<size_t>(codebook) * static_cast<size_t>(frames) + static_cast<size_t>(frame);
            if (src < frameMajor.size() && dst < result.size())
                result[dst] = frameMajor[src];
        }
    }
    return result;
}

std::vector<int32_t> codebookMajorToFrameMajor(const std::vector<int32_t>& codebookMajor, int codebooks, int frames)
{
    std::vector<int32_t> result;
    if (codebooks <= 0 || frames <= 0)
        return result;

    result.resize(static_cast<size_t>(codebooks * frames), 0);
    for (int codebook = 0; codebook < codebooks; ++codebook)
    {
        for (int frame = 0; frame < frames; ++frame)
        {
            const size_t src = static_cast<size_t>(codebook) * static_cast<size_t>(frames) + static_cast<size_t>(frame);
            const size_t dst = static_cast<size_t>(frame) * static_cast<size_t>(codebooks) + static_cast<size_t>(codebook);
            if (src < codebookMajor.size() && dst < result.size())
                result[dst] = codebookMajor[src];
        }
    }
    return result;
}
} // namespace

ModelBackendHttp::ModelBackendHttp(const juce::String& serverUrl)
    : serverUrl_(serverUrl)
{
    const juce::String timeoutEnv = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_HTTP_TIMEOUT_MS", {});
    if (timeoutEnv.isNotEmpty())
    {
        const int parsed = timeoutEnv.getIntValue();
        if (parsed > 0)
            timeoutMs_ = juce::jlimit(1000, 300000, parsed);
    }
}

ModelBackendHttp::~ModelBackendHttp() = default;

bool ModelBackendHttp::load(const std::string& modelRoot)
{
    juce::ignoreUnused(modelRoot);

    ready_.store(false);
    supportsPcmEndpoints_ = false;

    if (!checkHealth())
        return false;

    auto response = makeHttpRequest("/health", "GET");
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Failed to connect to server: " + response.error;
        return false;
    }

    auto healthObj = juce::JSON::parse(response.body);
    if (auto* obj = healthObj.getDynamicObject())
    {
        auto sampleRateVar = obj->getProperty("sample_rate");
        auto codebookCountVar = obj->getProperty("codebook_count");
        auto embeddingDimVar = obj->getProperty("embedding_dim");

        sampleRate_ = sampleRateVar.isVoid() ? 44100 : static_cast<int>(sampleRateVar);
        codebookCount_ = codebookCountVar.isVoid() ? 1 : static_cast<int>(codebookCountVar);
        embeddingDim_ = embeddingDimVar.isVoid() ? 128 : static_cast<int>(embeddingDimVar);
        requiredInputChannels_ = 1;
        frameRateHz_ = 0.0;
        tokenLayout_ = TokenLayout::CodebookMajor;
        activeCodec_ = "dac";

        // Capabilities endpoint is optional for backward compatibility.
        refreshCapabilities();

        ready_.store(true);
        juce::ScopedLock lock(errorMutex_);
        lastError_.clear();
        return true;
    }

    juce::ScopedLock lock(errorMutex_);
    lastError_ = "Invalid health response from server";
    return false;
}

bool ModelBackendHttp::setCodec(const std::string& codecId)
{
    if (codecId.empty())
        return false;

    juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
    requestObj->setProperty("codec", juce::String(codecId));
    juce::String jsonRequest = juce::JSON::toString(requestObj.get());

    auto response = makeHttpRequest("/codec", "POST", jsonRequest);
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Failed to set codec '" + juce::String(codecId) + "': " + response.error;
        return false;
    }

    if (!parseCapabilities(response.body))
    {
        // Old bridge may return a minimal ACK payload; refresh capabilities if possible.
        refreshCapabilities();
    }

    return true;
}

TokenBlock ModelBackendHttp::parseTokenBlockResponse(const juce::String& responseBody, bool* ok) const
{
    TokenBlock block;

    auto responseObj = juce::JSON::parse(responseBody);
    auto* obj = responseObj.getDynamicObject();
    if (obj == nullptr)
    {
        if (ok != nullptr)
            *ok = false;
        return block;
    }

    const int batch = static_cast<int>(getPropertyOrDefault(obj, "B", 1));
    const int frames = static_cast<int>(getPropertyOrDefault(obj, "T", 0));
    const int codebooks = static_cast<int>(getPropertyOrDefault(obj, "codebooks", codebookCount_));
    const juce::String layout = getPropertyOrDefault(obj, "token_layout", tokenLayoutToString(TokenLayout::CodebookMajor)).toString();

    auto* tokensArray = obj->getProperty("tokens").getArray();
    if (tokensArray == nullptr || frames <= 0 || codebooks <= 0)
    {
        if (ok != nullptr)
            *ok = false;
        return block;
    }

    std::vector<int32_t> rawTokens;
    rawTokens.reserve(static_cast<size_t>(tokensArray->size()));
    for (int i = 0; i < tokensArray->size(); ++i)
        rawTokens.push_back(static_cast<int32_t>(tokensArray->getUnchecked(i)));

    block.batchSize = batch;
    block.codebooks = codebooks;
    block.frames = frames;

    if (tokenLayoutFromString(layout) == TokenLayout::FrameMajor)
        block.tokens = frameMajorToCodebookMajor(rawTokens, codebooks, frames);
    else
        block.tokens = std::move(rawTokens);

    if (ok != nullptr)
        *ok = static_cast<int>(block.tokens.size()) == block.codebooks * block.frames;
    return block;
}

TokenBlock ModelBackendHttp::encodePCM(const juce::AudioBuffer<float>& audioBuffer)
{
    if (!ready_.load() || audioBuffer.getNumChannels() == 0 || audioBuffer.getNumSamples() == 0)
        return {};

    if (supportsPcmEndpoints_)
    {
        const int channels = audioBuffer.getNumChannels();
        const int samples = audioBuffer.getNumSamples();
        juce::MemoryBlock interleaved;
        interleaved.setSize(static_cast<size_t>(channels * samples) * sizeof(float), false);

        float* dst = static_cast<float*>(interleaved.getData());
        for (int sample = 0; sample < samples; ++sample)
            for (int channel = 0; channel < channels; ++channel)
                dst[static_cast<size_t>(sample) * static_cast<size_t>(channels) + static_cast<size_t>(channel)] =
                    audioBuffer.getSample(channel, sample);

        juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
        requestObj->setProperty("sample_rate", sampleRate_);
        requestObj->setProperty("channels", channels);
        requestObj->setProperty("num_samples", samples);
        requestObj->setProperty("dtype", "float32le");
        requestObj->setProperty("pcm_layout", "interleaved");
        requestObj->setProperty("pcm_b64", encodeBase64(interleaved));

        auto response = makeHttpRequest("/encode_pcm", "POST", juce::JSON::toString(requestObj.get()));
        if (response.success)
        {
            bool ok = false;
            TokenBlock block = parseTokenBlockResponse(response.body, &ok);
            if (ok)
                return block;
        }
    }

    // Legacy fallback endpoint (/encode) for backward compatibility.
    juce::TemporaryFile tempFile(".wav");
    std::unique_ptr<juce::OutputStream> outputStream = tempFile.getFile().createOutputStream();
    if (outputStream == nullptr)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Failed to create temporary file output stream";
        return {};
    }

    juce::WavAudioFormat wavFormat;
    std::unique_ptr<juce::AudioFormatWriter> writer = wavFormat.createWriterFor(
        outputStream,
        juce::AudioFormatWriterOptions()
            .withSampleRate(sampleRate_)
            .withNumChannels(audioBuffer.getNumChannels())
            .withBitsPerSample(16));

    if (writer == nullptr)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Failed to create temporary audio file";
        return {};
    }

    writer->writeFromAudioSampleBuffer(audioBuffer, 0, audioBuffer.getNumSamples());
    writer.reset();

    juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
    requestObj->setProperty("path", tempFile.getFile().getFullPathName());

    auto response = makeHttpRequest("/encode", "POST", juce::JSON::toString(requestObj.get()));
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Encode request failed: " + response.error;
        return {};
    }

    bool ok = false;
    TokenBlock block = parseTokenBlockResponse(response.body, &ok);
    if (!ok)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Invalid encode response format";
        return {};
    }

    return block;
}

std::vector<float> ModelBackendHttp::tokensToVectorRow(const TokenBlock& block, int frameIndex)
{
    if (!ready_.load() || frameIndex < 0 || frameIndex >= block.frames || block.tokens.empty())
        return {};

    juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
    requestObj->setProperty("token_layout", tokenLayoutToString(TokenLayout::CodebookMajor));

    juce::Array<juce::var> tokensArray;
    if (block.batchSize == 1)
    {
        // Fast path: send only the requested frame to reduce payload size and JSON parsing overhead.
        requestObj->setProperty("B", 1);
        requestObj->setProperty("T", 1);
        requestObj->setProperty("codebooks", block.codebooks);
        for (int codebook = 0; codebook < block.codebooks; ++codebook)
            tokensArray.add(block.tokens[block.index(codebook, frameIndex)]);
        requestObj->setProperty("tokens", tokensArray);
        requestObj->setProperty("frame_index", 0);
    }
    else
    {
        requestObj->setProperty("B", block.batchSize);
        requestObj->setProperty("T", block.frames);
        requestObj->setProperty("codebooks", block.codebooks);
        for (const auto token : block.tokens)
            tokensArray.add(token);
        requestObj->setProperty("tokens", tokensArray);
        requestObj->setProperty("frame_index", frameIndex);
    }

    auto response = makeHttpRequest("/tokens_to_vectors", "POST", juce::JSON::toString(requestObj.get()));
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "tokens_to_vectors request failed: " + response.error;
        return {};
    }

    auto responseObj = juce::JSON::parse(response.body);
    if (auto* obj = responseObj.getDynamicObject())
    {
        auto* vectorArray = obj->getProperty("vector").getArray();
        if (vectorArray != nullptr)
        {
            std::vector<float> result;
            result.reserve(static_cast<size_t>(vectorArray->size()));

            for (int i = 0; i < vectorArray->size(); ++i)
                result.push_back(static_cast<float>(vectorArray->getUnchecked(i)));

            return result;
        }
    }

    juce::ScopedLock lock(errorMutex_);
    lastError_ = "Invalid tokens_to_vectors response";
    return {};
}

bool ModelBackendHttp::tokensToVectorRows(const TokenBlock& block,
                                          int startFrame,
                                          int frameCount,
                                          std::vector<std::vector<float>>& out)
{
    out.clear();
    if (!ready_.load() || frameCount <= 0 || startFrame < 0 || startFrame >= block.frames || block.tokens.empty())
        return false;

    juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
    requestObj->setProperty("B", block.batchSize);
    requestObj->setProperty("T", block.frames);
    requestObj->setProperty("codebooks", block.codebooks);
    requestObj->setProperty("token_layout", tokenLayoutToString(TokenLayout::CodebookMajor));
    requestObj->setProperty("start_frame", startFrame);
    requestObj->setProperty("frame_count", frameCount);

    juce::Array<juce::var> tokensArray;
    for (const auto token : block.tokens)
        tokensArray.add(token);
    requestObj->setProperty("tokens", tokensArray);

    auto response = makeHttpRequest("/tokens_to_vectors_batch", "POST", juce::JSON::toString(requestObj.get()));
    if (response.success)
    {
        auto responseObj = juce::JSON::parse(response.body);
        if (auto* obj = responseObj.getDynamicObject())
        {
            auto* vectorsArray = obj->getProperty("vectors").getArray();
            if (vectorsArray != nullptr && vectorsArray->size() > 0)
            {
                out.reserve(static_cast<size_t>(vectorsArray->size()));
                for (int i = 0; i < vectorsArray->size(); ++i)
                {
                    auto* rowArray = vectorsArray->getUnchecked(i).getArray();
                    if (rowArray == nullptr || rowArray->size() <= 0)
                    {
                        out.clear();
                        break;
                    }

                    std::vector<float> row;
                    row.reserve(static_cast<size_t>(rowArray->size()));
                    for (int j = 0; j < rowArray->size(); ++j)
                        row.push_back(static_cast<float>(rowArray->getUnchecked(j)));
                    out.push_back(std::move(row));
                }

                if (!out.empty())
                    return true;
            }
        }
    }

    // Fallback for older bridge versions without batch endpoint support.
    return ModelBackend::tokensToVectorRows(block, startFrame, frameCount, out);
}

juce::AudioBuffer<float> ModelBackendHttp::decodeTokens(const TokenBlock& block)
{
    if (!ready_.load() || block.tokens.empty())
        return {};

    if (supportsPcmEndpoints_)
    {
        juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
        requestObj->setProperty("B", block.batchSize);
        requestObj->setProperty("T", block.frames);
        requestObj->setProperty("codebooks", block.codebooks);
        requestObj->setProperty("token_layout", tokenLayoutToString(TokenLayout::CodebookMajor));

        juce::Array<juce::var> tokensArray;
        for (const auto token : block.tokens)
            tokensArray.add(token);
        requestObj->setProperty("tokens", tokensArray);

        auto response = makeHttpRequest("/decode_pcm", "POST", juce::JSON::toString(requestObj.get()));
        if (response.success)
        {
            auto responseObj = juce::JSON::parse(response.body);
            if (auto* obj = responseObj.getDynamicObject())
            {
                const juce::String pcmBase64 = obj->getProperty("pcm_b64").toString();
                if (pcmBase64.isNotEmpty())
                {
                    const int channels = static_cast<int>(getPropertyOrDefault(obj, "channels", 1));
                    int numSamples = static_cast<int>(getPropertyOrDefault(obj, "num_samples", 0));
                    const auto pcmBytes = decodeBase64(pcmBase64);

                    const int totalFloats = static_cast<int>(pcmBytes.getSize() / sizeof(float));
                    if (numSamples <= 0 && channels > 0)
                        numSamples = totalFloats / channels;

                    if (channels > 0 && numSamples > 0 && totalFloats >= channels * numSamples)
                    {
                        juce::AudioBuffer<float> buffer(channels, numSamples);
                        const auto* src = static_cast<const float*>(pcmBytes.getData());
                        for (int sample = 0; sample < numSamples; ++sample)
                        {
                            for (int channel = 0; channel < channels; ++channel)
                            {
                                const size_t idx = static_cast<size_t>(sample) * static_cast<size_t>(channels) + static_cast<size_t>(channel);
                                buffer.setSample(channel, sample, src[idx]);
                            }
                        }
                        return buffer;
                    }
                }
            }
        }
    }

    // Legacy fallback endpoint (/decode) returning wav_b64.
    juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
    requestObj->setProperty("B", block.batchSize);
    requestObj->setProperty("T", block.frames);
    requestObj->setProperty("codebooks", block.codebooks);

    juce::Array<juce::var> tokensArray;
    for (const auto token : block.tokens)
        tokensArray.add(token);
    requestObj->setProperty("tokens", tokensArray);

    auto response = makeHttpRequest("/decode", "POST", juce::JSON::toString(requestObj.get()));
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Decode request failed: " + response.error;
        return {};
    }

    auto responseObj = juce::JSON::parse(response.body);
    if (auto* obj = responseObj.getDynamicObject())
    {
        const juce::String wavBase64 = obj->getProperty("wav_b64").toString();
        if (wavBase64.isNotEmpty())
        {
            auto audioData = decodeBase64(wavBase64);
            juce::MemoryInputStream memStream(audioData, false);

            juce::WavAudioFormat wavFormat;
            std::unique_ptr<juce::AudioFormatReader> reader(wavFormat.createReaderFor(&memStream, false));

            if (reader != nullptr)
            {
                juce::AudioBuffer<float> buffer(static_cast<int>(reader->numChannels),
                                                static_cast<int>(reader->lengthInSamples));
                reader->read(&buffer, 0, static_cast<int>(reader->lengthInSamples), 0, true, true);
                return buffer;
            }
        }
    }

    juce::ScopedLock lock(errorMutex_);
    lastError_ = "Invalid decode response";
    return {};
}

void ModelBackendHttp::setServerUrl(const juce::String& url)
{
    serverUrl_ = url;
    ready_.store(false);
}

juce::String ModelBackendHttp::getServerUrl() const
{
    return serverUrl_;
}

bool ModelBackendHttp::checkHealth()
{
    auto response = makeHttpRequest("/health", "GET");
    if (response.success)
    {
        auto healthObj = juce::JSON::parse(response.body);
        if (auto* obj = healthObj.getDynamicObject())
        {
            auto okVar = obj->getProperty("ok");
            const bool ok = okVar.isVoid() ? false : static_cast<bool>(okVar);
            return ok;
        }
    }

    juce::ScopedLock lock(errorMutex_);
    lastError_ = "Health check failed: " + response.error;
    return false;
}

bool ModelBackendHttp::refreshCapabilities()
{
    auto response = makeHttpRequest("/capabilities", "GET");
    if (!response.success)
        return false;

    return parseCapabilities(response.body);
}

juce::String ModelBackendHttp::activeCodec() const
{
    return activeCodec_;
}

bool ModelBackendHttp::parseCapabilities(const juce::String& responseBody)
{
    auto parsed = juce::JSON::parse(responseBody);
    auto* obj = parsed.getDynamicObject();
    if (obj == nullptr)
        return false;

    sampleRate_ = static_cast<int>(getPropertyOrDefault(obj, "sample_rate", sampleRate_));
    codebookCount_ = static_cast<int>(getPropertyOrDefault(obj, "codebook_count", codebookCount_));
    embeddingDim_ = static_cast<int>(getPropertyOrDefault(obj, "embedding_dim", embeddingDim_));
    requiredInputChannels_ = static_cast<int>(getPropertyOrDefault(obj, "required_input_channels", requiredInputChannels_));
    frameRateHz_ = static_cast<double>(getPropertyOrDefault(obj, "frame_rate_hz", frameRateHz_));
    activeCodec_ = getPropertyOrDefault(obj, "active_codec", activeCodec_).toString();
    tokenLayout_ = tokenLayoutFromString(getPropertyOrDefault(obj, "token_layout", tokenLayoutToString(tokenLayout_)).toString());

    const auto supportsPcmVar = obj->getProperty("supports_pcm_endpoints");
    if (!supportsPcmVar.isVoid())
        supportsPcmEndpoints_ = static_cast<bool>(supportsPcmVar);

    supportedCodecs_.clear();
    if (auto* codecsArray = obj->getProperty("supported_codecs").getArray())
    {
        supportedCodecs_.reserve(static_cast<size_t>(codecsArray->size()));
        for (int i = 0; i < codecsArray->size(); ++i)
            supportedCodecs_.push_back(codecsArray->getUnchecked(i).toString());
    }

    return true;
}

juce::String ModelBackendHttp::getLastError() const
{
    juce::ScopedLock lock(errorMutex_);
    return lastError_;
}

ModelBackendHttp::HttpResponse ModelBackendHttp::makeHttpRequest(const juce::String& endpoint,
                                                                 const juce::String& method,
                                                                 const juce::String& jsonBody) const
{
    HttpResponse result;

    juce::URL url(serverUrl_ + endpoint);
    if (method == "POST" && jsonBody.isNotEmpty())
        url = url.withPOSTData(jsonBody);

    juce::URL::InputStreamOptions options(juce::URL::ParameterHandling::inAddress);
    auto withTimeout = options.withConnectionTimeoutMs(timeoutMs_);
    auto withHeaders = jsonBody.isNotEmpty()
                           ? withTimeout.withExtraHeaders("Content-Type: application/json")
                           : withTimeout;
    auto streamOptions = method.isNotEmpty() ? withHeaders.withHttpRequestCmd(method) : withHeaders;

    std::unique_ptr<juce::InputStream> stream(url.createInputStream(streamOptions));

    if (stream != nullptr)
    {
        result.success = true;
        result.statusCode = 200;
        result.body = stream->readEntireStreamAsString();
    }
    else
    {
        result.success = false;
        result.statusCode = 0;
        result.error = "Failed to connect to " + url.toString(true);
    }

    return result;
}

juce::String ModelBackendHttp::encodeBase64(const juce::MemoryBlock& data) const
{
    return juce::Base64::toBase64(data.getData(), data.getSize());
}

juce::MemoryBlock ModelBackendHttp::decodeBase64(const juce::String& encoded) const
{
    juce::MemoryBlock result;
    juce::MemoryOutputStream stream(result, false);
    juce::Base64::convertFromBase64(stream, encoded);
    return result;
}

std::unique_ptr<ModelBackend> createHttpModelBackend(const juce::String& serverUrl)
{
    return std::make_unique<ModelBackendHttp>(serverUrl);
}

#endif // NM_WITH_PYBRIDGE
