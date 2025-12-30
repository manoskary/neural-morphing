#include "ModelBackendHttp.h"

#if NM_WITH_PYBRIDGE

#include "JuceHeader.h"

ModelBackendHttp::ModelBackendHttp(const juce::String& serverUrl)
    : serverUrl_(serverUrl)
{
}

ModelBackendHttp::~ModelBackendHttp() = default;

bool ModelBackendHttp::load(const std::string& modelRoot)
{
    juce::ignoreUnused(modelRoot);
    
    // Check if server is healthy and get configuration
    if (!checkHealth())
        return false;

    // Get server configuration from health endpoint
    auto response = makeHttpRequest("/health", "GET");
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Failed to connect to server: " + response.error;
        return false;
    }

    // Parse health response to get server configuration
    auto healthObj = juce::JSON::parse(response.body);
    if (auto* obj = healthObj.getDynamicObject())
    {
        auto sampleRateVar = obj->getProperty("sample_rate");
        auto codebookCountVar = obj->getProperty("codebook_count");
        auto embeddingDimVar = obj->getProperty("embedding_dim");
        
        sampleRate_ = sampleRateVar.isVoid() ? 44100 : static_cast<int>(sampleRateVar);
        codebookCount_ = codebookCountVar.isVoid() ? 1 : static_cast<int>(codebookCountVar);
        embeddingDim_ = embeddingDimVar.isVoid() ? 128 : static_cast<int>(embeddingDimVar);
        
        ready_.store(true);
        
        juce::ScopedLock lock(errorMutex_);
        lastError_ = juce::String();
        return true;
    }
    
    juce::ScopedLock lock(errorMutex_);
    lastError_ = "Invalid health response from server";
    return false;
}

TokenBlock ModelBackendHttp::encodePCM(const juce::AudioBuffer<float>& mono)
{
    if (!ready_.load() || mono.getNumChannels() == 0 || mono.getNumSamples() == 0)
        return {};

    // Save audio to temporary file
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
            .withNumChannels(mono.getNumChannels())
            .withBitsPerSample(16));

    if (writer == nullptr)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Failed to create temporary audio file";
        return {};
    }

    writer->writeFromAudioSampleBuffer(mono, 0, mono.getNumSamples());
    writer.reset();

    // Create JSON request
    juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
    requestObj->setProperty("path", tempFile.getFile().getFullPathName());
    
    juce::String jsonRequest = juce::JSON::toString(requestObj.get());

    // Make HTTP request
    auto response = makeHttpRequest("/encode", "POST", jsonRequest);
    
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Encode request failed: " + response.error;
        return {};
    }

    // Parse response
    auto responseObj = juce::JSON::parse(response.body);
    if (auto* obj = responseObj.getDynamicObject())
    {
        auto BVar = obj->getProperty("B");
        auto TVar = obj->getProperty("T");
        auto codebooksVar = obj->getProperty("codebooks");
        
        int B = BVar.isVoid() ? 1 : static_cast<int>(BVar);
        int T = TVar.isVoid() ? 0 : static_cast<int>(TVar);
        int codebooks = codebooksVar.isVoid() ? codebookCount_ : static_cast<int>(codebooksVar);
        auto* tokensArray = obj->getProperty("tokens").getArray();

        if (tokensArray == nullptr || T == 0)
        {
            juce::ScopedLock lock(errorMutex_);
            lastError_ = "Invalid encode response format";
            return {};
        }

        TokenBlock block;
        block.batchSize = B;
        block.codebooks = codebooks;
        block.frames = T;
        block.tokens.reserve(static_cast<size_t>(block.codebooks * T));

        if (codebookCount_ == 0 && codebooks > 0)
            codebookCount_ = codebooks;

        for (int i = 0; i < tokensArray->size(); ++i)
            block.tokens.push_back(static_cast<int32_t>(tokensArray->getUnchecked(i)));

        return block;
    }

    juce::ScopedLock lock(errorMutex_);
    lastError_ = "Invalid encode response";
    return {};
}

std::vector<float> ModelBackendHttp::tokensToVectorRow(const TokenBlock& block, int frameIndex)
{
    if (!ready_.load() || frameIndex < 0 || frameIndex >= block.frames || block.tokens.empty())
        return {};

    // Create JSON request
    juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
    requestObj->setProperty("B", block.batchSize);
    requestObj->setProperty("T", block.frames);
    
    juce::Array<juce::var> tokensArray;
    for (size_t i = 0; i < block.tokens.size(); ++i)
        tokensArray.add(block.tokens[i]);
    requestObj->setProperty("tokens", tokensArray);
    requestObj->setProperty("frame_index", frameIndex);
    
    juce::String jsonRequest = juce::JSON::toString(requestObj.get());

    // Make HTTP request
    auto response = makeHttpRequest("/tokens_to_vectors", "POST", jsonRequest);
    
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "tokens_to_vectors request failed: " + response.error;
        return {};
    }

    // Parse response
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

juce::AudioBuffer<float> ModelBackendHttp::decodeTokens(const TokenBlock& block)
{
    if (!ready_.load() || block.tokens.empty())
        return {};

    // Create JSON request
    juce::DynamicObject::Ptr requestObj = new juce::DynamicObject();
    requestObj->setProperty("B", block.batchSize);
    requestObj->setProperty("T", block.frames);
    
    juce::Array<juce::var> tokensArray;
    for (size_t i = 0; i < block.tokens.size(); ++i)
        tokensArray.add(block.tokens[i]);
    requestObj->setProperty("tokens", tokensArray);
    
    juce::String jsonRequest = juce::JSON::toString(requestObj.get());

    // Make HTTP request
    auto response = makeHttpRequest("/decode", "POST", jsonRequest);
    
    if (!response.success)
    {
        juce::ScopedLock lock(errorMutex_);
        lastError_ = "Decode request failed: " + response.error;
        return {};
    }

    // Parse response
    auto responseObj = juce::JSON::parse(response.body);
    if (auto* obj = responseObj.getDynamicObject())
    {
        auto wavBase64Var = obj->getProperty("wav_b64");
        juce::String wavBase64 = wavBase64Var.isVoid() ? juce::String() : wavBase64Var.toString();
        if (wavBase64.isNotEmpty())
        {
            // Decode base64 to binary data
            auto audioData = decodeBase64(wavBase64);
            
            // Create memory input stream from binary data
            juce::MemoryInputStream memStream(audioData, false);
            
            // Read audio from stream
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
            bool ok = okVar.isVoid() ? false : static_cast<bool>(okVar);
            return ok;
        }
    }
    
    juce::ScopedLock lock(errorMutex_);
    lastError_ = "Health check failed: " + response.error;
    return false;
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
    
    // For POST requests with body data, add it to the URL
    if (method == "POST" && jsonBody.isNotEmpty())
    {
        url = url.withPOSTData(jsonBody);
    }
    
    // Create URL options for the request
    juce::URL::InputStreamOptions options(juce::URL::ParameterHandling::inAddress);
    auto withTimeout = options.withConnectionTimeoutMs(timeoutMs_);
    auto withHeaders = jsonBody.isNotEmpty()
                            ? withTimeout.withExtraHeaders("Content-Type: application/json")
                            : withTimeout;
    auto streamOptions = method.isNotEmpty() ? withHeaders.withHttpRequestCmd(method) : withHeaders;

    // Make the request
    std::unique_ptr<juce::InputStream> stream(url.createInputStream(streamOptions));
    
    if (stream != nullptr)
    {
        result.success = true;
        result.statusCode = 200; // JUCE URL doesn't provide status code directly
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
