#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "JuceHeader.h"
#include <cmath>
#if NM_HAS_ONNX
#include "ModelBackendOnnx.h"
#endif
#if NM_WITH_PYBRIDGE
#include "ModelBackendHttp.h"
#endif

namespace
{
constexpr int monoScratchReserve = 8192;
}

NeuralMorphingAudioProcessor::NeuralMorphingAudioProcessor()
    : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, juce::Identifier("PARAMETERS"), createParameterLayout()),
      decodedFifo_(64)
{
    initialiseBackend();
    createWorkers();

    monoScratch_.setSize(1, monoScratchReserve);
}

NeuralMorphingAudioProcessor::~NeuralMorphingAudioProcessor()
{
    shutdownWorkers();
}

const juce::String NeuralMorphingAudioProcessor::getName() const
{
    return ProjectInfo::projectName;
}

void NeuralMorphingAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate_ = sampleRate;
    samplesPerBlock_ = samplesPerBlock;

    onsetDetector_.prepare(sampleRate, 512, 256);
    onsetDetector_.reset();
    resetMorphSmoothing();

    isPrepared_ = true;

    if (paletteWorker_ != nullptr && !paletteWorker_->isThreadRunning())
        paletteWorker_->startThread();

    setLatencySamples(2048);
}

void NeuralMorphingAudioProcessor::releaseResources()
{
    isPrepared_ = false;
}

bool NeuralMorphingAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

void NeuralMorphingAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused(midi);

    const int totalNumInputChannels = getTotalNumInputChannels();
    const int totalNumOutputChannels = getTotalNumOutputChannels();
    const int numSamples = buffer.getNumSamples();

    for (int channel = totalNumInputChannels; channel < totalNumOutputChannels; ++channel)
        buffer.clear(channel, 0, numSamples);

    if (resetSmoothingPending_.exchange(false, std::memory_order_acq_rel))
        resetMorphSmoothing();

    if (isStandaloneWrapper())
        renderStandaloneSource(buffer);

    juce::AudioBuffer<float> dryBuffer;
    dryBuffer.makeCopyOf(buffer);

    const bool backendReady = backend_ != nullptr && backend_->ready();

    if (backendReady)
    {
        if (monoScratch_.getNumSamples() < numSamples)
            monoScratch_.setSize(1, numSamples, false, false, true);

        monoScratch_.clear();
        const int channels = juce::jmax(1, buffer.getNumChannels());
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            monoScratch_.addFrom(0, 0, buffer, ch, 0, numSamples, 1.0f / static_cast<float>(channels));

        if (!paletteReady())
        {
            if (isSilent(monoScratch_))
                targetSegments_.clear();
            resetMorphSmoothing();
        }
        else
        {
            if (!isSilent(monoScratch_))
            {
                auto targetTokens = backend_->encodePCM(monoScratch_);
                if (!targetTokens.tokens.empty() && targetTokens.frames > 0)
                {
                    if (targetSegments_.size() >= maxTargetSegments_)
                        targetSegments_.erase(targetSegments_.begin());

                    targetSegments_.push_back(targetTokens);

                    const uint64_t cacheKey = hashTokenBlock(targetTokens);
                    juce::AudioBuffer<float> cachedAudio;
                    bool hasCachedAudio = false;
                    {
                        const juce::SpinLock::ScopedLockType lock(morphCacheMutex_);
                        for (size_t i = 0; i < morphCache_.size(); ++i)
                        {
                            if (morphCache_[i].hash == cacheKey)
                            {
                                if (i > 0)
                                {
                                    auto entry = std::move(morphCache_[i]);
                                    morphCache_.erase(morphCache_.begin() + static_cast<long>(i));
                                    morphCache_.insert(morphCache_.begin(), std::move(entry));
                                }
                                if (morphCache_.front().audio.getNumSamples() > 0)
                                {
                                    cachedAudio.makeCopyOf(morphCache_.front().audio);
                                    hasCachedAudio = true;
                                }
                                break;
                            }
                        }
                    }

                    if (hasCachedAudio)
                    {
                        mixMorphedAudio(buffer, dryBuffer, cachedAudio);
                    }
                    else
                    {
                        auto matchedTokens = buildMatchedTokenBlock(targetTokens);
                        if (!matchedTokens.tokens.empty() && matchedTokens.frames > 0)
                        {
                            auto morphedAudio = backend_->decodeTokens(matchedTokens);
                            mixMorphedAudio(buffer, dryBuffer, morphedAudio);

                            MorphCacheEntry entry;
                            entry.hash = cacheKey;
                            entry.matchedTokens = std::move(matchedTokens);
                            entry.audio = std::move(morphedAudio);
                            {
                                const juce::SpinLock::ScopedLockType lock(morphCacheMutex_);
                                morphCache_.insert(morphCache_.begin(), std::move(entry));

                                if (morphCache_.size() > maxMorphCacheEntries_)
                                    morphCache_.pop_back();
                            }
                        }
                    }
                }
            }
            else
            {
                targetSegments_.clear();
                resetMorphSmoothing();
            }
        }
    }

    const float outputGain = juce::Decibels::decibelsToGain(getParam("outputGain"));
    buffer.applyGain(outputGain);
}

juce::AudioProcessorEditor* NeuralMorphingAudioProcessor::createEditor()
{
    return new NeuralMorphingAudioProcessorEditor(*this);
}

void NeuralMorphingAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    xml->setAttribute("pluginName", "NeuralMorphingState");
    copyXmlToBinary(*xml, destData);
}

void NeuralMorphingAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml != nullptr)
    {
        if (xml->hasTagName("NeuralMorphingState"))
            parameters.replaceState(juce::ValueTree::fromXml(*xml));
    }
}

float NeuralMorphingAudioProcessor::getParam(const juce::String& paramID) const
{
    if (auto* value = parameters.getRawParameterValue(paramID))
        return *value;
    return 0.0f;
}

void NeuralMorphingAudioProcessor::setStandaloneSource(juce::AudioBuffer<float> buffer, double sampleRate, const juce::String& name)
{
    const juce::ScopedLock lock(standaloneMutex_);
    standaloneSourceBuffer_ = std::move(buffer);
    standaloneSourceSampleRate_ = sampleRate;
    standaloneSourcePosition_ = 0;
    standaloneSourceName_ = name;
    standaloneSourceLoaded_ = standaloneSourceBuffer_.getNumSamples() > 0;
}

void NeuralMorphingAudioProcessor::clearStandaloneSource()
{
    const juce::ScopedLock lock(standaloneMutex_);
    standaloneSourceBuffer_.setSize(0, 0);
    standaloneSourceSampleRate_ = 0.0;
    standaloneSourcePosition_ = 0;
    standaloneSourceName_.clear();
    standaloneSourceLoaded_ = false;
}

bool NeuralMorphingAudioProcessor::hasStandaloneSource() const
{
    const juce::ScopedLock lock(standaloneMutex_);
    return standaloneSourceLoaded_;
}

juce::String NeuralMorphingAudioProcessor::standaloneSourceName() const
{
    const juce::ScopedLock lock(standaloneMutex_);
    return standaloneSourceName_;
}

void NeuralMorphingAudioProcessor::invalidateMorphCache()
{
    const juce::SpinLock::ScopedLockType lock(morphCacheMutex_);
    morphCache_.clear();
    resetSmoothingPending_.store(true, std::memory_order_release);
}

bool NeuralMorphingAudioProcessor::isStandaloneWrapper() const
{
    return wrapperType == juce::AudioProcessor::wrapperType_Standalone;
}

bool NeuralMorphingAudioProcessor::renderStandaloneSource(juce::AudioBuffer<float>& buffer)
{
    const juce::ScopedLock lock(standaloneMutex_);
    if (!standaloneSourceLoaded_ || standaloneSourceBuffer_.getNumSamples() == 0)
        return false;

    const int totalSamples = standaloneSourceBuffer_.getNumSamples();
    if (standaloneSourcePosition_ >= totalSamples)
    {
        buffer.clear();
        return true;
    }

    const int numSamples = buffer.getNumSamples();
    const int outputChannels = buffer.getNumChannels();
    const int sourceChannels = standaloneSourceBuffer_.getNumChannels();
    const int samplesRemaining = juce::jmax(0, totalSamples - static_cast<int>(standaloneSourcePosition_));
    const int samplesToCopy = juce::jmin(numSamples, samplesRemaining);

    buffer.clear();
    for (int ch = 0; ch < outputChannels; ++ch)
    {
        const int srcCh = juce::jmin(ch, sourceChannels - 1);
        buffer.copyFrom(ch, 0, standaloneSourceBuffer_, srcCh, static_cast<int>(standaloneSourcePosition_), samplesToCopy);
    }

    standaloneSourcePosition_ += samplesToCopy;
    return true;
}

void NeuralMorphingAudioProcessor::resetMorphSmoothing()
{
    morphSmoothingState_.assign(static_cast<size_t>(juce::jmax(1, getTotalNumOutputChannels())), 0.0f);
}

uint64_t NeuralMorphingAudioProcessor::hashTokenBlock(const TokenBlock& block) const
{
    constexpr uint64_t fnvOffset = 1469598103934665603ULL;
    constexpr uint64_t fnvPrime = 1099511628211ULL;
    uint64_t hash = fnvOffset;

    auto mix = [&](uint64_t value)
    {
        hash ^= value;
        hash *= fnvPrime;
    };

    mix(static_cast<uint64_t>(block.frames));
    mix(static_cast<uint64_t>(block.codebooks));
    mix(static_cast<uint64_t>(block.tokens.size()));

    for (const auto token : block.tokens)
        mix(static_cast<uint64_t>(static_cast<uint32_t>(token)));

    return hash;
}

juce::AudioProcessorValueTreeState::ParameterLayout NeuralMorphingAudioProcessor::createParameterLayout()
{
    using R = juce::NormalisableRange<float>;
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>("temperature", "Temperature", R(0.1f, 2.0f, 0.01f), 1.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("threshold", "Threshold", R(0.1f, 2.0f, 0.01f), 1.0f));
    params.push_back(std::make_unique<juce::AudioParameterInt>("unit", "Unit", 1, 10, 2));
    params.push_back(std::make_unique<juce::AudioParameterInt>("stride", "Stride", 1, 10, 2));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("similarity", "Similarity", R(0.0f, 1.0f, 0.01f), 0.8f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("envelopeFollow", "Envelope Follow", R(0.0f, 1.0f, 0.01f), 0.7f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("dryWet", "Dry/Wet", R(0.0f, 1.0f, 0.01f), 1.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("outputGain", "Output Gain (dB)", R(-24.0f, 24.0f, 0.1f), 0.0f));

    // Backend selection (0=Native/ONNX, 1=Python Bridge)
    juce::StringArray backendChoices;
    backendChoices.add("Native");
#if NM_WITH_PYBRIDGE
    backendChoices.add("Python Bridge");
#endif
    params.push_back(std::make_unique<juce::AudioParameterChoice>("backend", "Backend", backendChoices, 0));

    return { params.begin(), params.end() };
}

void NeuralMorphingAudioProcessor::refreshBackendSampleRate(double sampleRate)
{
    juce::ignoreUnused(sampleRate);
}

void NeuralMorphingAudioProcessor::mixWetBuffer(juce::AudioBuffer<float>& buffer, juce::AudioBuffer<float>& dryBuffer)
{
    juce::ignoreUnused(buffer, dryBuffer);
}

TokenBlock NeuralMorphingAudioProcessor::buildMatchedTokenBlock(const TokenBlock& targetBlock)
{
    TokenBlock result;

    if (!paletteReady())
        return result;

    result.batchSize = targetBlock.batchSize;
    result.codebooks = targetBlock.codebooks;
    result.frames = targetBlock.frames;
    result.tokens.resize(static_cast<std::size_t>(result.codebooks * result.frames));

    for (int frame = 0; frame < targetBlock.frames; ++frame)
    {
        auto targetVector = backend_->tokensToVectorRow(targetBlock, frame);
        if (targetVector.empty())
        {
            for (int cb = 0; cb < result.codebooks; ++cb)
            {
                const int idx = targetBlock.index(cb, frame);
                result.tokens[static_cast<std::size_t>(result.index(cb, frame))] =
                    (idx >= 0 && static_cast<std::size_t>(idx) < targetBlock.tokens.size())
                        ? targetBlock.tokens[static_cast<std::size_t>(idx)]
                        : 0;
            }
            continue;
        }

        auto matches = paletteIndex_->query(targetVector, 1);
        if (matches.empty())
        {
            for (int cb = 0; cb < result.codebooks; ++cb)
            {
                const int idx = targetBlock.index(cb, frame);
                result.tokens[static_cast<std::size_t>(result.index(cb, frame))] =
                    (idx >= 0 && static_cast<std::size_t>(idx) < targetBlock.tokens.size())
                        ? targetBlock.tokens[static_cast<std::size_t>(idx)]
                        : 0;
            }
            continue;
        }

        const auto& best = matches.front();
        const auto& bestMeta = paletteIndex_->meta(best.annIndex);
        const TokenBlock* sourceBlock = paletteIndex_->tokensForMeta(bestMeta);

        for (int cb = 0; cb < result.codebooks; ++cb)
        {
            const int destIdx = result.index(cb, frame);

            int tokenValue = 0;
            if (sourceBlock != nullptr && bestMeta.frame < sourceBlock->frames)
            {
                const int srcIdx = sourceBlock->index(cb, bestMeta.frame);
                if (srcIdx >= 0 && static_cast<std::size_t>(srcIdx) < sourceBlock->tokens.size())
                    tokenValue = sourceBlock->tokens[static_cast<std::size_t>(srcIdx)];
            }
            else
            {
                const int fallbackIdx = targetBlock.index(cb, frame);
                if (fallbackIdx >= 0 && static_cast<std::size_t>(fallbackIdx) < targetBlock.tokens.size())
                    tokenValue = targetBlock.tokens[static_cast<std::size_t>(fallbackIdx)];
            }

            result.tokens[static_cast<std::size_t>(destIdx)] = tokenValue;
        }
    }

    return result;
}

void NeuralMorphingAudioProcessor::mixMorphedAudio(juce::AudioBuffer<float>& buffer,
                                                   const juce::AudioBuffer<float>& dryBuffer,
                                                   const juce::AudioBuffer<float>& morphed)
{
    const int totalNumOutputChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    if (morphed.getNumChannels() == 0 || morphed.getNumSamples() == 0)
        return;

    const float similarity = juce::jlimit(0.0f, 1.0f, getParam("similarity"));
    const float dryWet = juce::jlimit(0.0f, 1.0f, getParam("dryWet"));
    const float smoothingAmount = juce::jlimit(0.0f, 1.0f, getParam("envelopeFollow"));

    if (smoothingAmount <= 0.0f)
        resetMorphSmoothing();

    float smoothingAlpha = 0.0f;
    const bool useSmoothing = smoothingAmount > 0.0f && currentSampleRate_ > 0.0;
    if (useSmoothing)
    {
        const float smoothingMs = 2.0f + smoothingAmount * 18.0f;
        smoothingAlpha = std::exp(-1.0f / (0.001f * smoothingMs * static_cast<float>(currentSampleRate_)));
    }

    if (morphSmoothingState_.size() < static_cast<size_t>(totalNumOutputChannels))
        morphSmoothingState_.assign(static_cast<size_t>(totalNumOutputChannels), 0.0f);

    const int copySamples = juce::jmin(numSamples, morphed.getNumSamples());

    for (int ch = 0; ch < totalNumOutputChannels; ++ch)
    {
        const int morphCh = juce::jmin(ch, morphed.getNumChannels() - 1);
        const int dryCh = juce::jmin(ch, dryBuffer.getNumChannels() - 1);
        float prevSmoothed = morphSmoothingState_[static_cast<size_t>(ch)];

        for (int sample = 0; sample < copySamples; ++sample)
        {
            const float drySample = dryBuffer.getSample(dryCh, sample);
            float morphSample = morphed.getSample(morphCh, sample);
            if (useSmoothing)
            {
                morphSample = (1.0f - smoothingAlpha) * morphSample + smoothingAlpha * prevSmoothed;
                prevSmoothed = morphSample;
            }
            const float blendedMorph = similarity * morphSample + (1.0f - similarity) * drySample;
            const float outputSample = dryWet * blendedMorph + (1.0f - dryWet) * drySample;
            buffer.setSample(ch, sample, outputSample);
        }

        if (useSmoothing)
            morphSmoothingState_[static_cast<size_t>(ch)] = prevSmoothed;

        for (int sample = copySamples; sample < numSamples; ++sample)
        {
            const float drySample = dryBuffer.getSample(dryCh, sample);
            const float outputSample = dryWet * drySample + (1.0f - dryWet) * drySample;
            buffer.setSample(ch, sample, outputSample);
        }
    }
}

bool NeuralMorphingAudioProcessor::paletteReady() const
{
    if (paletteIndex_ == nullptr)
        return false;

    return paletteIndex_->size() > 0;
}

bool NeuralMorphingAudioProcessor::isSilent(const juce::AudioBuffer<float>& buffer) const
{
    const int numSamples = buffer.getNumSamples();
    if (numSamples == 0)
        return true;

    double sumSquares = 0.0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        const auto* data = buffer.getReadPointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            const float sample = data[i];
            sumSquares += static_cast<double>(sample) * static_cast<double>(sample);
        }
    }

    const double rms = std::sqrt(sumSquares / static_cast<double>(buffer.getNumChannels() * juce::jmax(1, numSamples)));
    return rms < 1.0e-6;
}

void NeuralMorphingAudioProcessor::shutdownWorkers()
{
    if (paletteWorker_ != nullptr)
    {
        paletteWorker_->shutdown();

        if (paletteWorker_->isThreadRunning())
        {
            paletteWorker_->signalThreadShouldExit();
            paletteWorker_->stopThread(2000);
        }
        paletteWorker_.reset();
    }

    if (matchWorker_ != nullptr)
    {
        matchWorker_->shutdown();

        if (matchWorker_->isThreadRunning())
        {
            matchWorker_->signalThreadShouldExit();
            matchWorker_->stopThread(2000);
        }
        matchWorker_.reset();
    }

    decodedFifo_.clear();
}

void NeuralMorphingAudioProcessor::createWorkers()
{
    if (backend_ == nullptr || paletteIndex_ == nullptr)
        return;

    paletteWorker_ = std::make_unique<PaletteWorker>(*backend_, *paletteIndex_);
    matchWorker_.reset();

    if (isPrepared_)
    {
        if (!paletteWorker_->isThreadRunning())
            paletteWorker_->startThread();
    }
}

void NeuralMorphingAudioProcessor::initialiseBackend()
{
    // Get backend selection from parameters
    int backendChoice = 0;
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("backend")))
        backendChoice = param->getIndex();

    // Try to create the selected backend
    if (backendChoice == 1) // Python Bridge
    {
#if NM_WITH_PYBRIDGE
        juce::String serverUrl = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_SERVER_URL", "http://localhost:8000");
        auto httpBackend = createHttpModelBackend(serverUrl);
        if (httpBackend != nullptr && httpBackend->load(""))
        {
            backend_ = std::move(httpBackend);
        }
#endif
    }
    else // Native backend (ONNX or stub)
    {
#if NM_HAS_ONNX
        if (backend_ == nullptr)
        {
            const juce::String modelRoot = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_MODEL_DIR", {});
            if (modelRoot.isNotEmpty())
            {
                auto onnxBackend = createOnnxModelBackend();
                if (onnxBackend != nullptr && onnxBackend->load(modelRoot.toStdString()))
                    backend_ = std::move(onnxBackend);
            }
        }
#endif
    }

    // Fallback to stub backend if nothing else worked
    if (backend_ == nullptr)
        backend_ = createStubModelBackend();

    if (backend_ != nullptr && !backend_->ready())
        backend_->load("");

    const int vectorDim = (backend_ != nullptr) ? juce::jmax(1, backend_->embeddingDimension()) : 2;
    paletteIndex_ = std::make_unique<PaletteIndex>(vectorDim);
}

void NeuralMorphingAudioProcessor::switchBackend(int backendType)
{
    // Set the backend parameter
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("backend")))
        *param = backendType;

    // Clear current backend
    shutdownWorkers();
    backend_.reset();
    paletteIndex_.reset();
    invalidateMorphCache();

    // Reinitialize with new backend
    initialiseBackend();
    createWorkers();
}

juce::String NeuralMorphingAudioProcessor::getBackendStatus() const
{
    if (backend_ == nullptr)
        return "No backend loaded";
        
    if (!backend_->ready())
        return "Backend not ready";

#if NM_WITH_PYBRIDGE
    if (auto* httpBackend = dynamic_cast<ModelBackendHttp*>(backend_.get()))
    {
        juce::String error = httpBackend->getLastError();
        if (error.isNotEmpty())
            return "HTTP Backend Error: " + error;
        return "HTTP Backend connected to " + httpBackend->getServerUrl();
    }
#endif

    return "Backend ready (" + juce::String(backend_->sampleRate()) + " Hz, " 
           + juce::String(backend_->codebookCount()) + " codebooks)";
}

bool NeuralMorphingAudioProcessor::isBackendReady() const
{
    return backend_ != nullptr && backend_->ready();
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralMorphingAudioProcessor();
}
