#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "JuceHeader.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#if NM_HAS_ONNX
#include "ModelBackendOnnx.h"
#endif
#if NM_WITH_PYBRIDGE
#include "ModelBackendHttp.h"
#endif

namespace
{
constexpr int monoScratchReserve = 8192;
constexpr int matchCandidateCount = 96;
constexpr int matchBeamWidth = 12;
constexpr int topKMixCount = 4;

struct RvqGroups
{
    int q0 = 0;
    int q1 = 0;
    int codebooks = 0;
    int dimsPerCodebook = 0;
};

RvqGroups makeRvqGroups(int codebooks, int embeddingDim)
{
    RvqGroups groups;
    groups.codebooks = codebooks;

    if (codebooks <= 0 || embeddingDim <= 0 || embeddingDim % codebooks != 0)
    {
        groups.q0 = codebooks;
        groups.q1 = codebooks;
        groups.dimsPerCodebook = 0;
        return groups;
    }

    groups.dimsPerCodebook = embeddingDim / codebooks;
    groups.q0 = std::max(1, codebooks / 3);
    groups.q1 = std::max(groups.q0 + 1, (2 * codebooks) / 3);
    groups.q1 = std::min(groups.q1, codebooks);
    return groups;
}

float cosineDistanceSlice(const std::vector<float>& a, const std::vector<float>& b, int start, int length)
{
    if (length <= 0)
        return 0.0f;

    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (int i = 0; i < length; ++i)
    {
        const float va = a[static_cast<size_t>(start + i)];
        const float vb = b[static_cast<size_t>(start + i)];
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }

    const float denom = std::sqrt(normA) * std::sqrt(normB) + 1.0e-9f;
    return 1.0f - dot / denom;
}

int getEnvIntClamped(const char* key, int fallback, int minValue, int maxValue)
{
    const char* value = std::getenv(key);
    if (value == nullptr)
        return fallback;

    const auto parsed = juce::String(value).getIntValue();
    return juce::jlimit(minValue, maxValue, parsed);
}
}

NeuralMorphingAudioProcessor::NeuralMorphingAudioProcessor()
    : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, juce::Identifier("PARAMETERS"), createParameterLayout()),
      decodedFifo_(64)
{
    initialiseBackend();
    configureRealtimeTimings();
    createWorkers();

    backendInputScratch_.setSize(2, monoScratchReserve);
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
    configureRealtimeTimings();
    morphUpdateCountdownSamples_ = 0;
    hasLastRealtimeMorphBlock_ = false;
    lastRealtimeMorphBlock_.setSize(0, 0);

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

void NeuralMorphingAudioProcessor::configureRealtimeTimings()
{
    const bool heavyRealtimeBackend = backend_ != nullptr
                                      && backend_->requiredInputChannels() >= 2
                                      && backend_->codebookCount() >= 32
                                      && backend_->frameRateHz() <= 30.0f;

    morphUpdateIntervalMs_ = heavyRealtimeBackend ? 80 : 40;
    morphUpdateIntervalMs_ = getEnvIntClamped("NEURAL_MORPHING_RT_UPDATE_MS", morphUpdateIntervalMs_, 10, 500);

    realtimeEncodeWindowMs_ = heavyRealtimeBackend ? 85 : 0;
    realtimeEncodeWindowMs_ = getEnvIntClamped("NEURAL_MORPHING_RT_ENCODE_WINDOW_MS", realtimeEncodeWindowMs_, 0, 2000);

    const int blockSamples = juce::jmax(1, samplesPerBlock_);
    realtimeEncodeWindowSamples_ = blockSamples;
    if (realtimeEncodeWindowMs_ > 0 && currentSampleRate_ > 0.0)
    {
        realtimeEncodeWindowSamples_ = juce::jmax(
            blockSamples,
            static_cast<int>(currentSampleRate_ * (static_cast<double>(realtimeEncodeWindowMs_) / 1000.0)));
    }

    const int fallbackInputChannels = juce::jmax(1, getTotalNumInputChannels());
    const int requiredChannels = juce::jmax(1, backend_ != nullptr ? backend_->requiredInputChannels() : fallbackInputChannels);
    if (realtimeInputHistory_.getNumChannels() != requiredChannels
        || realtimeInputHistory_.getNumSamples() != realtimeEncodeWindowSamples_)
    {
        realtimeInputHistory_.setSize(requiredChannels, realtimeEncodeWindowSamples_, false, false, true);
    }
    realtimeInputHistory_.clear();
    realtimeInputFilledSamples_ = 0;
}

void NeuralMorphingAudioProcessor::pushRealtimeInputHistory(const juce::AudioBuffer<float>& inputBlock)
{
    const int historySamples = realtimeInputHistory_.getNumSamples();
    if (historySamples <= 0 || inputBlock.getNumSamples() <= 0)
        return;

    const int channels = juce::jmin(realtimeInputHistory_.getNumChannels(), inputBlock.getNumChannels());
    const int copySamples = juce::jmin(inputBlock.getNumSamples(), historySamples);
    const int shiftSamples = historySamples - copySamples;
    const int inputOffset = inputBlock.getNumSamples() - copySamples;

    for (int ch = 0; ch < channels; ++ch)
    {
        float* dst = realtimeInputHistory_.getWritePointer(ch);
        if (shiftSamples > 0)
            juce::FloatVectorOperations::copy(dst, dst + copySamples, shiftSamples);

        const float* src = inputBlock.getReadPointer(ch, inputOffset);
        juce::FloatVectorOperations::copy(dst + shiftSamples, src, copySamples);
    }

    realtimeInputFilledSamples_ = juce::jmin(historySamples, realtimeInputFilledSamples_ + copySamples);
}

const juce::AudioBuffer<float>& NeuralMorphingAudioProcessor::selectRealtimeEncodeInput(
    const juce::AudioBuffer<float>& currentBlock) const
{
    if (realtimeEncodeWindowSamples_ > currentBlock.getNumSamples()
        && realtimeInputHistory_.getNumSamples() >= realtimeEncodeWindowSamples_
        && realtimeInputFilledSamples_ >= realtimeEncodeWindowSamples_)
    {
        return realtimeInputHistory_;
    }
    return currentBlock;
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
    const float dryWet = juce::jlimit(0.0f, 1.0f, getParam("dryWet"));

    // Hard bypass: if fully dry, avoid bridge/model work in the realtime callback.
    if (dryWet <= 0.0f)
    {
        targetSegments_.clear();
        hasLastRealtimeMorphBlock_ = false;
        morphUpdateCountdownSamples_ = 0;
        realtimeInputHistory_.clear();
        realtimeInputFilledSamples_ = 0;
        resetMorphSmoothing();
        lastMatchedIndex_ = -1;

        const float outputGain = juce::Decibels::decibelsToGain(getParam("outputGain"));
        buffer.applyGain(outputGain);
        return;
    }

    const bool backendReady = backend_ != nullptr && backend_->ready();

    if (backendReady)
    {
        const int requiredChannels = juce::jmax(1, backend_->requiredInputChannels());
        if (backendInputScratch_.getNumSamples() < numSamples || backendInputScratch_.getNumChannels() != requiredChannels)
            backendInputScratch_.setSize(requiredChannels, numSamples, false, false, true);

        backendInputScratch_.clear();
        const int inputChannels = juce::jmax(1, buffer.getNumChannels());

        if (requiredChannels == 1)
        {
            for (int ch = 0; ch < inputChannels; ++ch)
                backendInputScratch_.addFrom(0, 0, buffer, ch, 0, numSamples, 1.0f / static_cast<float>(inputChannels));
        }
        else
        {
            for (int ch = 0; ch < requiredChannels; ++ch)
            {
                const int srcCh = juce::jmin(ch, inputChannels - 1);
                backendInputScratch_.copyFrom(ch, 0, buffer, srcCh, 0, numSamples);
            }
        }
        pushRealtimeInputHistory(backendInputScratch_);

        if (!paletteReady())
        {
            if (isSilent(backendInputScratch_))
                targetSegments_.clear();
            hasLastRealtimeMorphBlock_ = false;
            realtimeInputHistory_.clear();
            realtimeInputFilledSamples_ = 0;
            resetMorphSmoothing();
            lastMatchedIndex_ = -1;
            morphUpdateCountdownSamples_ = 0;
        }
        else
        {
            if (!isSilent(backendInputScratch_))
            {
                bool shouldUpdateMorph = true;
                if (morphUpdateCountdownSamples_ > 0)
                {
                    morphUpdateCountdownSamples_ = juce::jmax(0, morphUpdateCountdownSamples_ - numSamples);
                    shouldUpdateMorph = false;
                }
                else if (currentSampleRate_ > 0.0)
                {
                    morphUpdateCountdownSamples_ = juce::jmax(
                        1,
                        static_cast<int>(currentSampleRate_ * (static_cast<double>(morphUpdateIntervalMs_) / 1000.0)));
                }

                if (!shouldUpdateMorph)
                {
                    if (hasLastRealtimeMorphBlock_ && lastRealtimeMorphBlock_.getNumSamples() > 0)
                        mixMorphedAudio(buffer, dryBuffer, lastRealtimeMorphBlock_);
                    const float outputGain = juce::Decibels::decibelsToGain(getParam("outputGain"));
                    buffer.applyGain(outputGain);
                    return;
                }

                const auto& encodeInput = selectRealtimeEncodeInput(backendInputScratch_);
                auto targetTokens = backend_->encodePCM(encodeInput);
                if (!targetTokens.tokens.empty() && targetTokens.frames > 0)
                {
                    if (targetSegments_.size() >= maxTargetSegments_)
                        targetSegments_.erase(targetSegments_.begin());

                    targetSegments_.push_back(targetTokens);

                    const uint64_t cacheKey = hashMorphKey(targetTokens);
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
                        lastRealtimeMorphBlock_.makeCopyOf(cachedAudio);
                        hasLastRealtimeMorphBlock_ = true;
                    }
                    else
                    {
                        auto matchedTokens = buildMatchedTokenBlock(targetTokens);
                        if (!matchedTokens.tokens.empty() && matchedTokens.frames > 0)
                        {
                            auto morphedAudio = backend_->decodeTokens(matchedTokens);
                            mixMorphedAudio(buffer, dryBuffer, morphedAudio);
                            lastRealtimeMorphBlock_.makeCopyOf(morphedAudio);
                            hasLastRealtimeMorphBlock_ = true;

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
                hasLastRealtimeMorphBlock_ = false;
                realtimeInputHistory_.clear();
                realtimeInputFilledSamples_ = 0;
                resetMorphSmoothing();
                lastMatchedIndex_ = -1;
                morphUpdateCountdownSamples_ = 0;
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
    hasLastRealtimeMorphBlock_ = false;
    lastRealtimeMorphBlock_.setSize(0, 0);
    morphUpdateCountdownSamples_ = 0;
    realtimeInputHistory_.clear();
    realtimeInputFilledSamples_ = 0;
    lastMatchedIndex_ = -1;
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
    const int numSamples = buffer.getNumSamples();
    const int outputChannels = buffer.getNumChannels();
    const int sourceChannels = standaloneSourceBuffer_.getNumChannels();
    buffer.clear();

    int writePosition = 0;
    while (writePosition < numSamples)
    {
        if (standaloneSourcePosition_ >= totalSamples)
            standaloneSourcePosition_ = 0;

        const int sourceOffset = static_cast<int>(standaloneSourcePosition_);
        const int samplesRemainingInSource = juce::jmax(0, totalSamples - sourceOffset);
        const int samplesToCopy = juce::jmin(numSamples - writePosition, samplesRemainingInSource);
        if (samplesToCopy <= 0)
            break;

        for (int ch = 0; ch < outputChannels; ++ch)
        {
            const int srcCh = juce::jmin(ch, sourceChannels - 1);
            buffer.copyFrom(ch, writePosition, standaloneSourceBuffer_, srcCh, sourceOffset, samplesToCopy);
        }

        writePosition += samplesToCopy;
        standaloneSourcePosition_ += samplesToCopy;
    }

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

uint64_t NeuralMorphingAudioProcessor::hashMorphKey(const TokenBlock& block) const
{
    uint64_t hash = hashTokenBlock(block);

    auto mix = [&](uint64_t value)
    {
        constexpr uint64_t fnvPrime = 1099511628211ULL;
        hash ^= value;
        hash *= fnvPrime;
    };

    mix(static_cast<uint64_t>(getParam("unit")));
    mix(static_cast<uint64_t>(getParam("stride")));
    mix(static_cast<uint64_t>(std::lround(getParam("temperature") * 1000.0f)));
    mix(static_cast<uint64_t>(std::lround(getParam("threshold") * 1000.0f)));
    mix(static_cast<uint64_t>(std::lround(getParam("continuity") * 1000.0f)));
    mix(static_cast<uint64_t>(std::lround(getParam("rvqFocus") * 1000.0f)));

    return hash;
}

juce::AudioProcessorValueTreeState::ParameterLayout NeuralMorphingAudioProcessor::createParameterLayout()
{
    using R = juce::NormalisableRange<float>;
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>("temperature", "Temperature", R(0.1f, 2.0f, 0.01f), 1.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("threshold", "Threshold", R(0.1f, 2.0f, 0.01f), 1.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("continuity", "Continuity", R(0.0f, 1.0f, 0.01f), 0.3f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("rvqFocus", "RVQ Focus", R(0.0f, 1.0f, 0.01f), 0.5f));
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

    // Bridge codec selection (used by HTTP backend, ignored by native backends)
    juce::StringArray bridgeCodecChoices;
    bridgeCodecChoices.add("DAC");
    bridgeCodecChoices.add("SpectroStream");
    params.push_back(std::make_unique<juce::AudioParameterChoice>("bridgeCodec", "Bridge Codec", bridgeCodecChoices, 0));

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

    if (!paletteReady() || backend_ == nullptr)
        return result;

    if (targetBlock.tokens.empty() || targetBlock.frames <= 0)
        return result;

    result = targetBlock;

    const int unit = juce::jmax(1, static_cast<int>(getParam("unit")));
    const int stride = juce::jmax(1, static_cast<int>(getParam("stride")));

    if (targetBlock.frames < unit)
        return result;

    const int codebooks = targetBlock.codebooks;
    const int embeddingDim = backend_->embeddingDimension();
    RvqGroups groups = makeRvqGroups(codebooks, embeddingDim);

    int coarseLen = groups.q0 * groups.dimsPerCodebook;
    int midLen = (groups.q1 - groups.q0) * groups.dimsPerCodebook;
    int fineLen = (groups.codebooks - groups.q1) * groups.dimsPerCodebook;

    if (groups.dimsPerCodebook <= 0)
    {
        coarseLen = embeddingDim;
        midLen = 0;
        fineLen = 0;
        groups.q0 = codebooks;
        groups.q1 = codebooks;
    }

    float rvqFocus = juce::jlimit(0.0f, 1.0f, getParam("rvqFocus"));
    float weightCoarse = 1.0f - rvqFocus;
    float weightFine = rvqFocus;
    float weightMid = 0.5f * (weightCoarse + weightFine);
    if (coarseLen == 0)
        weightCoarse = 0.0f;
    if (midLen == 0)
        weightMid = 0.0f;
    if (fineLen == 0)
        weightFine = 0.0f;

    const float weightSum = weightCoarse + weightMid + weightFine;
    if (weightSum > 0.0f)
    {
        weightCoarse /= weightSum;
        weightMid /= weightSum;
        weightFine /= weightSum;
    }

    const float threshold = getParam("threshold");
    const float temperature = juce::jmax(1.0e-4f, getParam("temperature"));
    const float continuity = juce::jlimit(0.0f, 1.0f, getParam("continuity"));

    const int candidateCount = std::min(matchCandidateCount, paletteIndex_->size());
    if (candidateCount <= 0)
        return result;

    const int beamWidth = std::min(matchBeamWidth, candidateCount);

    struct CandidateInfo
    {
        int annIndex = -1;
        float emission = 0.0f;
        float fineDistance = 0.0f;
    };

    struct GrainMatch
    {
        int startFrame = 0;
        std::vector<CandidateInfo> candidates;
    };

    std::vector<GrainMatch> grains;
    grains.reserve(static_cast<size_t>((targetBlock.frames - unit) / stride + 1));

    std::vector<float> queryVector;
    queryVector.reserve(static_cast<size_t>(embeddingDim));
    std::vector<std::vector<float>> frameVectors;
    const bool hasFrameVectors = backend_->tokensToVectorRows(targetBlock, 0, targetBlock.frames, frameVectors)
                                 && static_cast<int>(frameVectors.size()) == targetBlock.frames;

    auto computeDescriptor = [&](int startFrame, std::vector<float>& out) -> bool
    {
        out.assign(static_cast<size_t>(embeddingDim), 0.0f);
        for (int offset = 0; offset < unit; ++offset)
        {
            std::vector<float> row;
            if (hasFrameVectors)
                row = frameVectors[static_cast<size_t>(startFrame + offset)];
            else
                row = backend_->tokensToVectorRow(targetBlock, startFrame + offset);
            if (static_cast<int>(row.size()) != embeddingDim)
                return false;

            for (int d = 0; d < embeddingDim; ++d)
                out[static_cast<size_t>(d)] += row[static_cast<size_t>(d)];
        }

        const float invUnit = 1.0f / static_cast<float>(unit);
        for (auto& value : out)
            value *= invUnit;

        return true;
    };

    for (int startFrame = 0; startFrame + unit <= targetBlock.frames; startFrame += stride)
    {
        if (!computeDescriptor(startFrame, queryVector))
            continue;

        auto matches = paletteIndex_->query(queryVector, candidateCount);
        if (matches.empty())
            continue;

        GrainMatch grain;
        grain.startFrame = startFrame;
        grain.candidates.reserve(matches.size());

        for (const auto& match : matches)
        {
            std::vector<float> candidateVector;
            if (!paletteIndex_->getVector(match.annIndex, candidateVector))
                continue;
            if (static_cast<int>(candidateVector.size()) != embeddingDim)
                continue;

            const float distCoarse = (coarseLen > 0) ? cosineDistanceSlice(queryVector, candidateVector, 0, coarseLen) : 0.0f;
            const float distMid = (midLen > 0) ? cosineDistanceSlice(queryVector, candidateVector, coarseLen, midLen) : 0.0f;
            const float distFine = (fineLen > 0) ? cosineDistanceSlice(queryVector, candidateVector, coarseLen + midLen, fineLen) : 0.0f;
            const float emission = weightCoarse * distCoarse + weightMid * distMid + weightFine * distFine;
            const float fineDistance = (fineLen > 0) ? distFine : emission;

            grain.candidates.push_back({ match.annIndex, emission, fineDistance });
        }

        if (grain.candidates.empty())
            continue;

        std::sort(grain.candidates.begin(), grain.candidates.end(),
                  [](const CandidateInfo& a, const CandidateInfo& b) { return a.emission < b.emission; });

        grains.push_back(std::move(grain));
    }

    if (grains.empty())
        return result;

    auto transitionCost = [&](int prevIndex, int nextIndex) -> float
    {
        const auto& prevMeta = paletteIndex_->meta(prevIndex);
        const auto& nextMeta = paletteIndex_->meta(nextIndex);
        float metaPenalty = 0.0f;
        if (prevMeta.sampleId != nextMeta.sampleId)
        {
            metaPenalty = 1.0f;
        }
        else
        {
            const int frameDelta = std::abs(nextMeta.frame - prevMeta.frame);
            if (frameDelta > stride)
                metaPenalty = std::min(1.0f, static_cast<float>(frameDelta - stride) / static_cast<float>(unit * 4));
        }

        const float latentPenalty = paletteIndex_->cosineDistance(prevIndex, nextIndex);
        return latentPenalty + metaPenalty;
    };

    struct BeamState
    {
        float score = 0.0f;
        int candidateIdx = -1;
        int back = -1;
    };

    std::vector<std::vector<BeamState>> beamHistory;
    beamHistory.reserve(grains.size());

    std::vector<BeamState> beam;
    beam.reserve(grains.front().candidates.size());

    for (size_t i = 0; i < grains.front().candidates.size(); ++i)
    {
        const auto& candidate = grains.front().candidates[i];
        float score = candidate.emission;
        if (continuity > 0.0f && lastMatchedIndex_ >= 0)
            score += continuity * transitionCost(lastMatchedIndex_, candidate.annIndex);
        beam.push_back({ score, static_cast<int>(i), -1 });
    }

    if (static_cast<int>(beam.size()) > beamWidth)
    {
        std::partial_sort(beam.begin(), beam.begin() + beamWidth, beam.end(),
                          [](const BeamState& a, const BeamState& b) { return a.score < b.score; });
        beam.resize(static_cast<size_t>(beamWidth));
    }
    beamHistory.push_back(beam);

    for (size_t g = 1; g < grains.size(); ++g)
    {
        const auto& prevGrain = grains[g - 1];
        const auto& grain = grains[g];
        std::vector<BeamState> nextBeam;
        nextBeam.reserve(beamHistory.back().size() * grain.candidates.size());

        for (size_t prevIdx = 0; prevIdx < beamHistory.back().size(); ++prevIdx)
        {
            const auto& prevState = beamHistory.back()[prevIdx];
            const int prevAnn = prevGrain.candidates[static_cast<size_t>(prevState.candidateIdx)].annIndex;

            for (size_t candIdx = 0; candIdx < grain.candidates.size(); ++candIdx)
            {
                const auto& candidate = grain.candidates[candIdx];
                float score = prevState.score + candidate.emission;
                if (continuity > 0.0f)
                    score += continuity * transitionCost(prevAnn, candidate.annIndex);
                nextBeam.push_back({ score, static_cast<int>(candIdx), static_cast<int>(prevIdx) });
            }
        }

        if (static_cast<int>(nextBeam.size()) > beamWidth)
        {
            std::partial_sort(nextBeam.begin(), nextBeam.begin() + beamWidth, nextBeam.end(),
                              [](const BeamState& a, const BeamState& b) { return a.score < b.score; });
            nextBeam.resize(static_cast<size_t>(beamWidth));
        }

        beamHistory.push_back(std::move(nextBeam));
    }

    std::vector<int> path(grains.size(), 0);
    if (!beamHistory.empty() && !beamHistory.back().empty())
    {
        int bestIdx = 0;
        for (size_t i = 1; i < beamHistory.back().size(); ++i)
        {
            if (beamHistory.back()[i].score < beamHistory.back()[static_cast<size_t>(bestIdx)].score)
                bestIdx = static_cast<int>(i);
        }

        for (int g = static_cast<int>(grains.size()) - 1; g >= 0; --g)
        {
            const auto& state = beamHistory[static_cast<size_t>(g)][static_cast<size_t>(bestIdx)];
            path[static_cast<size_t>(g)] = state.candidateIdx;
            bestIdx = state.back;
            if (bestIdx < 0)
                break;
        }
    }

    for (size_t g = 0; g < grains.size(); ++g)
    {
        auto& grain = grains[g];
        const int candidateIdx = path[g];
        if (candidateIdx < 0 || static_cast<size_t>(candidateIdx) >= grain.candidates.size())
            continue;

        const auto& best = grain.candidates[static_cast<size_t>(candidateIdx)];
        lastMatchedIndex_ = best.annIndex;
        const bool fallbackCoarse = best.emission > threshold;

        const int startFrame = grain.startFrame;
        const int span = juce::jmin(unit, result.frames - startFrame);
        if (span <= 0)
            continue;

        const int kCount = std::min(topKMixCount, static_cast<int>(grain.candidates.size()));
        std::vector<int> topIndices;
        std::vector<float> topDistances;
        topIndices.reserve(static_cast<size_t>(kCount));
        topDistances.reserve(static_cast<size_t>(kCount));

        for (int i = 0; i < kCount; ++i)
        {
            topIndices.push_back(grain.candidates[static_cast<size_t>(i)].annIndex);
            topDistances.push_back(grain.candidates[static_cast<size_t>(i)].fineDistance);
        }

        std::vector<float> weights;
        weights.resize(static_cast<size_t>(kCount), 1.0f / static_cast<float>(juce::jmax(1, kCount)));

        if (kCount > 1)
        {
            float maxLogit = -topDistances[0] / temperature;
            for (int i = 1; i < kCount; ++i)
                maxLogit = std::max(maxLogit, -topDistances[static_cast<size_t>(i)] / temperature);

            float sum = 0.0f;
            for (int i = 0; i < kCount; ++i)
            {
                const float logit = -topDistances[static_cast<size_t>(i)] / temperature;
                const float value = std::exp(logit - maxLogit);
                weights[static_cast<size_t>(i)] = value;
                sum += value;
            }

            if (sum > 0.0f)
            {
                for (auto& w : weights)
                    w /= sum;
            }
        }

        const auto& bestMeta = paletteIndex_->meta(best.annIndex);
        const TokenBlock* bestBlock = paletteIndex_->tokensForMeta(bestMeta);

        for (int q = 0; q < codebooks; ++q)
        {
            const bool isCoarse = q < groups.q0;
            const bool isMid = q >= groups.q0 && q < groups.q1;
            const bool isFine = q >= groups.q1;
            if (isCoarse && fallbackCoarse)
                continue;

            if (isFine)
            {
                for (int offset = 0; offset < span; ++offset)
                {
                    std::vector<int> codes;
                    codes.reserve(static_cast<size_t>(kCount));

                    for (int i = 0; i < kCount; ++i)
                    {
                        const auto& meta = paletteIndex_->meta(topIndices[static_cast<size_t>(i)]);
                        const TokenBlock* block = paletteIndex_->tokensForMeta(meta);
                        int tokenValue = 0;
                        if (block != nullptr && meta.frame + offset < block->frames)
                        {
                            const int idx = block->index(q, meta.frame + offset);
                            if (idx >= 0 && static_cast<size_t>(idx) < block->tokens.size())
                                tokenValue = block->tokens[static_cast<size_t>(idx)];
                        }
                        else
                        {
                            const int fallbackIdx = targetBlock.index(q, startFrame + offset);
                            if (fallbackIdx >= 0 && static_cast<size_t>(fallbackIdx) < targetBlock.tokens.size())
                                tokenValue = targetBlock.tokens[static_cast<size_t>(fallbackIdx)];
                        }
                        codes.push_back(tokenValue);
                    }

                    int bestCode = codes.front();
                    float bestScore = -1.0f;
                    for (int i = 0; i < kCount; ++i)
                    {
                        float score = 0.0f;
                        for (int j = 0; j < kCount; ++j)
                        {
                            if (codes[static_cast<size_t>(j)] == codes[static_cast<size_t>(i)])
                                score += weights[static_cast<size_t>(j)];
                        }

                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestCode = codes[static_cast<size_t>(i)];
                        }
                    }

                    result.tokens[static_cast<size_t>(result.index(q, startFrame + offset))] = bestCode;
                }
                continue;
            }

            if (!isCoarse && !isMid)
                continue;

            for (int offset = 0; offset < span; ++offset)
            {
                int tokenValue = 0;
                if (bestBlock != nullptr && bestMeta.frame + offset < bestBlock->frames)
                {
                    const int srcIdx = bestBlock->index(q, bestMeta.frame + offset);
                    if (srcIdx >= 0 && static_cast<size_t>(srcIdx) < bestBlock->tokens.size())
                        tokenValue = bestBlock->tokens[static_cast<size_t>(srcIdx)];
                }
                else
                {
                    const int fallbackIdx = targetBlock.index(q, startFrame + offset);
                    if (fallbackIdx >= 0 && static_cast<size_t>(fallbackIdx) < targetBlock.tokens.size())
                        tokenValue = targetBlock.tokens[static_cast<size_t>(fallbackIdx)];
                }

                result.tokens[static_cast<size_t>(result.index(q, startFrame + offset))] = tokenValue;
            }
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
            float outputSample = dryWet * blendedMorph + (1.0f - dryWet) * drySample;

            // Ensure Dry/Wet endpoints always reach true dry/true wet regardless of Similarity.
            if (dryWet <= 0.0f)
                outputSample = drySample;
            else if (dryWet >= 1.0f)
                outputSample = morphSample;

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

    if (paletteIndex_->size() <= 0)
        return false;

    const auto config = paletteIndex_->grainConfig();
    const int unit = juce::jmax(1, static_cast<int>(getParam("unit")));
    const int stride = juce::jmax(1, static_cast<int>(getParam("stride")));
    return config.unit == unit && config.stride == stride;
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

    const auto applyBridgeCodecSelection = [&]()
    {
        if (backend_ == nullptr)
            return;

        int codecChoice = 0;
        if (auto* codecParam = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("bridgeCodec")))
            codecChoice = codecParam->getIndex();

        const juce::String codecId = (codecChoice == 1) ? "spectrostream" : "dac";
        backend_->setCodec(codecId.toStdString());
    };

    // Fallback to stub backend if nothing else worked
    if (backend_ == nullptr)
        backend_ = createStubModelBackend();

    if (backend_ != nullptr && !backend_->ready())
        backend_->load("");

    applyBridgeCodecSelection();

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
    configureRealtimeTimings();
    createWorkers();
}

void NeuralMorphingAudioProcessor::setBridgeCodec(int codecType)
{
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("bridgeCodec")))
        *param = codecType;

    if (backend_ == nullptr)
        return;

    const juce::String codecId = (codecType == 1) ? "spectrostream" : "dac";
    const bool changed = backend_->setCodec(codecId.toStdString());
    if (changed)
    {
        // Codec switches can change embedding dimensionality (e.g. DAC vs SpectroStream),
        // so recreate the palette index/workers to keep vector dimensions aligned.
        shutdownWorkers();
        const int vectorDim = juce::jmax(1, backend_->embeddingDimension());
        paletteIndex_ = std::make_unique<PaletteIndex>(vectorDim);
        invalidateMorphCache();
        configureRealtimeTimings();
        createWorkers();
    }
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
        return "HTTP Backend connected to " + httpBackend->getServerUrl()
               + " | codec=" + httpBackend->activeCodec()
               + " | in=" + juce::String(backend_->requiredInputChannels()) + "ch";
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
