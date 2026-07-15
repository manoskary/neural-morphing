#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "JuceHeader.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
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
constexpr int topKMixCountDac = 7;
constexpr int topKMixCountSpectroStream = 8;
constexpr int kQualityLookaheadMsDefault = 1000;
constexpr int kQualitySuperframeSamplesDefault = 32768;
constexpr int kQualityHopSamplesDefault = 8192;
constexpr int kQualityCrossfadeSamplesDefault = 4096;
constexpr int kLiveCandidateCountDefault = 48;
constexpr int kLiveBeamWidthDefault = 6;

struct MorphDefaults
{
    float temperature = 0.47f;
    float threshold = 0.99f;
    float continuity = 0.10f;
    float rvqFocus = 0.30f;
    int unit = 7;
    int stride = 2;
    int swapMode = 2;
};

constexpr MorphDefaults kDacDefaults{};
constexpr MorphDefaults kSpectroStreamDefaults{
    0.4315337f,
    0.24313963f,
    0.7887727f,
    0.34608898f,
    2,
    2,
    0
};

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

int getEnvIntClamped(const char* key, int fallback, int minValue, int maxValue)
{
    const char* value = std::getenv(key);
    if (value == nullptr)
        return fallback;

    const auto parsed = juce::String(value).getIntValue();
    return juce::jlimit(minValue, maxValue, parsed);
}

NeuralMorphingAudioProcessor::ProcessingMode processingModeFromParam(int value)
{
    return value == 1 ? NeuralMorphingAudioProcessor::ProcessingMode::LiveRealtime
                      : NeuralMorphingAudioProcessor::ProcessingMode::QualityParity;
}

NeuralMorphingAudioProcessor::BackendPolicy backendPolicyFromParam(int value)
{
    if (value == 1)
        return NeuralMorphingAudioProcessor::BackendPolicy::BridgeOnly;
    if (value == 2)
        return NeuralMorphingAudioProcessor::BackendPolicy::NativeOnly;
    return NeuralMorphingAudioProcessor::BackendPolicy::NativePreferredBridgeFallback;
}

int tokenChangePercent(const TokenBlock& source, const TokenBlock& morphed)
{
    const auto count = std::min(source.tokens.size(), morphed.tokens.size());
    if (count == 0)
        return -1;

    size_t changed = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (source.tokens[i] != morphed.tokens[i])
            ++changed;
    }
    return static_cast<int>(std::lround(100.0 * static_cast<double>(changed) / static_cast<double>(count)));
}

uint32_t audioFingerprint(const juce::AudioBuffer<float>& audio)
{
    uint32_t hash = 2166136261u;
    for (int ch = 0; ch < audio.getNumChannels(); ++ch)
    {
        const auto* data = audio.getReadPointer(ch);
        for (int i = 0; i < audio.getNumSamples(); ++i)
        {
            const auto q = static_cast<uint32_t>(static_cast<int16_t>(
                std::lround(juce::jlimit(-1.0f, 1.0f, data[i]) * 32767.0f)));
            hash ^= q;
            hash *= 16777619u;
        }
    }
    return hash;
}

juce::AudioBuffer<float> resampleAudio(const juce::AudioBuffer<float>& input,
                                       double sourceSampleRate,
                                       double outputSampleRate)
{
    juce::AudioBuffer<float> output;
    if (input.getNumChannels() <= 0 || input.getNumSamples() <= 0)
        return output;

    if (sourceSampleRate <= 0.0 || outputSampleRate <= 0.0
        || std::abs(sourceSampleRate - outputSampleRate) < 1.0)
    {
        output.makeCopyOf(input, true);
        return output;
    }

    const double ratio = sourceSampleRate / outputSampleRate;
    const int outputSamples = juce::jmax(
        1,
        static_cast<int>(std::lround(static_cast<double>(input.getNumSamples()) / ratio)));
    output.setSize(input.getNumChannels(), outputSamples, false, false, true);
    output.clear();

    for (int channel = 0; channel < input.getNumChannels(); ++channel)
    {
        juce::LagrangeInterpolator interpolator;
        interpolator.process(ratio,
                             input.getReadPointer(channel),
                             output.getWritePointer(channel),
                             outputSamples);
    }
    return output;
}

float bufferRms(const juce::AudioBuffer<float>& buffer)
{
    if (buffer.getNumChannels() <= 0 || buffer.getNumSamples() <= 0)
        return 0.0f;

    double energy = 0.0;
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        const float* samples = buffer.getReadPointer(channel);
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            energy += static_cast<double>(samples[sample]) * static_cast<double>(samples[sample]);
    }

    const double count = static_cast<double>(buffer.getNumChannels()) * static_cast<double>(buffer.getNumSamples());
    return std::sqrt(static_cast<float>(energy / count));
}

void matchWetLoudness(const juce::AudioBuffer<float>& source, juce::AudioBuffer<float>& wet)
{
    const float sourceRms = bufferRms(source);
    const float wetRms = bufferRms(wet);
    if (sourceRms > 1.0e-4f && wetRms > 1.0e-4f)
    {
        float wetPeak = 0.0f;
        for (int channel = 0; channel < wet.getNumChannels(); ++channel)
            wetPeak = juce::jmax(wetPeak, wet.getMagnitude(channel, 0, wet.getNumSamples()));

        constexpr float targetPeak = 0.8912509f;
        const float peakSafeGain = wetPeak > 1.0e-5f ? targetPeak / wetPeak : 1.0f;
        const float maximumGain = juce::jmin(32.0f, peakSafeGain);
        wet.applyGain(juce::jlimit(0.25f, juce::jmax(0.25f, maximumGain), sourceRms / wetRms));
    }
}

void mixOfflineMorphedAudio(juce::AudioBuffer<float>& output,
                            const juce::AudioBuffer<float>& source,
                            const juce::AudioBuffer<float>& wet,
                            float dryWet,
                            float envelopeAmount,
                            double sampleRate)
{
    dryWet = juce::jlimit(0.0f, 1.0f, dryWet);
    envelopeAmount = juce::jlimit(0.0f, 1.0f, envelopeAmount);
    const float dryGain = 1.0f - dryWet;
    const float wetGain = dryWet;
    const float attack = sampleRate > 0.0 ? std::exp(-1.0f / (0.005f * static_cast<float>(sampleRate))) : 0.0f;
    const float release = sampleRate > 0.0 ? std::exp(-1.0f / (0.080f * static_cast<float>(sampleRate))) : 0.0f;

    for (int channel = 0; channel < output.getNumChannels(); ++channel)
    {
        const int sourceChannel = juce::jmin(channel, source.getNumChannels() - 1);
        const int wetChannel = juce::jmin(channel, wet.getNumChannels() - 1);
        float sourceEnvelope = 0.0f;
        float wetEnvelope = 0.0f;

        for (int sample = 0; sample < output.getNumSamples(); ++sample)
        {
            const float sourceSample = source.getSample(sourceChannel, sample);
            float wetSample = sample < wet.getNumSamples() ? wet.getSample(wetChannel, sample) : 0.0f;

            if (envelopeAmount > 0.0f)
            {
                const float sourceMagnitude = std::abs(sourceSample);
                const float wetMagnitude = std::abs(wetSample);
                const float sourceCoefficient = sourceMagnitude > sourceEnvelope ? attack : release;
                const float wetCoefficient = wetMagnitude > wetEnvelope ? attack : release;
                sourceEnvelope = sourceCoefficient * sourceEnvelope + (1.0f - sourceCoefficient) * sourceMagnitude;
                wetEnvelope = wetCoefficient * wetEnvelope + (1.0f - wetCoefficient) * wetMagnitude;
                const float envelopeGain = juce::jlimit(0.0f, 4.0f, sourceEnvelope / (wetEnvelope + 1.0e-4f));
                wetSample *= 1.0f + envelopeAmount * (envelopeGain - 1.0f);
            }

            output.setSample(channel, sample, dryGain * sourceSample + wetGain * wetSample);
        }
    }
}

void applyOfflineSafetyLimiter(juce::AudioBuffer<float>& buffer)
{
    float peak = 0.0f;
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        peak = juce::jmax(peak, buffer.getMagnitude(channel, 0, buffer.getNumSamples()));

    constexpr float targetPeak = 0.8912509f;
    if (peak > targetPeak)
        buffer.applyGain(targetPeak / peak);
}

constexpr const char* morphTokenParameterIds[] = {
    "temperature",
    "threshold",
    "continuity",
    "rvqFocus",
    "similarity",
    "unit",
    "stride",
    "swapMode",
    "processingMode"
};
}

NeuralMorphingAudioProcessor::NeuralMorphingAudioProcessor()
    : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, juce::Identifier("PARAMETERS"), createParameterLayout()),
      decodedFifo_(64)
{
    qualityLookaheadMs_ = getEnvIntClamped("NEURAL_MORPHING_QUALITY_LOOKAHEAD_MS", kQualityLookaheadMsDefault, 50, 4000);
    qualitySuperframeSamples_ = getEnvIntClamped("NEURAL_MORPHING_QUALITY_SUPERFRAME", kQualitySuperframeSamplesDefault, 2048, 262144);
    qualityHopSamples_ = getEnvIntClamped("NEURAL_MORPHING_QUALITY_HOP", kQualityHopSamplesDefault, 256, qualitySuperframeSamples_);
    qualityCrossfadeSamples_ = getEnvIntClamped("NEURAL_MORPHING_QUALITY_CROSSFADE", kQualityCrossfadeSamplesDefault, 0, qualitySuperframeSamples_ / 2);
    liveCandidateCount_ = getEnvIntClamped("NEURAL_MORPHING_LIVE_CANDIDATES", kLiveCandidateCountDefault, 4, matchCandidateCount);
    liveBeamWidth_ = getEnvIntClamped("NEURAL_MORPHING_LIVE_BEAM", kLiveBeamWidthDefault, 1, matchBeamWidth);
    deterministicDemoPlayback_ = juce::SystemStats::getEnvironmentVariable(
                                     "NEURAL_MORPHING_DEMO_DETERMINISTIC",
                                     "0")
                                     .getIntValue()
                                 != 0;

    applyDacDemoDefaults();
    for (const auto* parameterId : morphTokenParameterIds)
        parameters.addParameterListener(parameterId, this);

    initialiseBackend();
    configureRealtimeTimings();
    createWorkers();

    backendInputScratch_.setSize(2, monoScratchReserve);
}

NeuralMorphingAudioProcessor::~NeuralMorphingAudioProcessor()
{
    for (const auto* parameterId : morphTokenParameterIds)
        parameters.removeParameterListener(parameterId, this);

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
    lastRealtimeMorphReadPosition_ = 0;
    lastRealtimeMorphWasUnderrun_ = false;
    wetAvailabilityMix_ = 0.0f;
    outputSafetyGain_ = 1.0f;
    qualityTailValid_ = false;
    qualityTailBuffer_.setSize(0, 0);

    onsetDetector_.prepare(sampleRate, 512, 256);
    onsetDetector_.reset();
    resetMorphSmoothing();
    dryScratch_.setSize(juce::jmax(1, getTotalNumOutputChannels()), juce::jmax(1, samplesPerBlock), false, false, true);

    isPrepared_ = true;

    if (paletteWorker_ != nullptr && !paletteWorker_->isThreadRunning())
        paletteWorker_->startThread();

    currentLatencySamples_ = currentLatencySamplesForMode();
    setLatencySamples(currentLatencySamples_);
}

void NeuralMorphingAudioProcessor::releaseResources()
{
    isPrepared_ = false;
}

void NeuralMorphingAudioProcessor::configureRealtimeTimings()
{
    const auto mode = selectedProcessingMode();

    if (mode == ProcessingMode::QualityParity)
    {
        realtimeEncodeWindowMs_ = qualityLookaheadMs_;
        if (currentSampleRate_ > 0.0)
        {
            realtimeEncodeWindowMs_ = juce::jmax(
                realtimeEncodeWindowMs_,
                static_cast<int>(std::ceil(1000.0 * static_cast<double>(qualitySuperframeSamples_) / currentSampleRate_)));
        }

        if (currentSampleRate_ > 0.0)
            morphUpdateIntervalMs_ = juce::jmax(5, static_cast<int>(std::ceil(1000.0 * static_cast<double>(qualityHopSamples_) / currentSampleRate_)));
        else
            morphUpdateIntervalMs_ = 200;
    }
    else
    {
        const bool heavyRealtimeBackend = backend_ != nullptr
                                          && backend_->requiredInputChannels() >= 2
                                          && backend_->codebookCount() >= 32
                                          && backend_->frameRateHz() <= 30.0f;

        morphUpdateIntervalMs_ = heavyRealtimeBackend ? 80 : 40;
        realtimeEncodeWindowMs_ = heavyRealtimeBackend ? 85 : 0;
    }

    morphUpdateIntervalMs_ = getEnvIntClamped("NEURAL_MORPHING_RT_UPDATE_MS", morphUpdateIntervalMs_, 5, 2000);
    realtimeEncodeWindowMs_ = getEnvIntClamped("NEURAL_MORPHING_RT_ENCODE_WINDOW_MS", realtimeEncodeWindowMs_, 0, 4000);

    if (backend_ != nullptr && backend_->frameRateHz() > 0.0)
    {
        constexpr int maximumGrainFrames = 10;
        const int minimumGrainWindowMs = static_cast<int>(std::ceil(
            1000.0 * static_cast<double>(maximumGrainFrames + 1) / backend_->frameRateHz()));
        realtimeEncodeWindowMs_ = juce::jmax(realtimeEncodeWindowMs_, minimumGrainWindowMs);
    }

    const int blockSamples = juce::jmax(1, samplesPerBlock_);
    realtimeEncodeWindowSamples_ = blockSamples;
    if (realtimeEncodeWindowMs_ > 0 && currentSampleRate_ > 0.0)
    {
        realtimeEncodeWindowSamples_ = juce::jmax(
            blockSamples,
            static_cast<int>(currentSampleRate_ * (static_cast<double>(realtimeEncodeWindowMs_) / 1000.0)));
    }

    if (mode == ProcessingMode::QualityParity)
        realtimeEncodeWindowSamples_ = juce::jmax(realtimeEncodeWindowSamples_, qualitySuperframeSamples_);

    const int fallbackInputChannels = juce::jmax(1, getTotalNumInputChannels());
    const int requiredChannels = juce::jmax(1, backend_ != nullptr ? backend_->requiredInputChannels() : fallbackInputChannels);
    if (realtimeInputHistory_.getNumChannels() != requiredChannels
        || realtimeInputHistory_.getNumSamples() != realtimeEncodeWindowSamples_)
    {
        realtimeInputHistory_.setSize(requiredChannels, realtimeEncodeWindowSamples_, false, false, true);
    }
    realtimeInputHistory_.clear();
    realtimeInputFilledSamples_ = 0;

    const juce::ScopedLock taskLock(realtimeTaskMutex_);
    if (pendingRealtimeEncodeInput_.getNumChannels() != requiredChannels
        || pendingRealtimeEncodeInput_.getNumSamples() != realtimeEncodeWindowSamples_)
    {
        pendingRealtimeEncodeInput_.setSize(requiredChannels, realtimeEncodeWindowSamples_, false, false, true);
    }
    pendingRealtimeEncodeInput_.clear();
    hasPendingRealtimeTask_ = false;

    if (qualityTailBuffer_.getNumChannels() != requiredChannels
        || qualityTailBuffer_.getNumSamples() != qualityCrossfadeSamples_)
    {
        qualityTailBuffer_.setSize(requiredChannels, qualityCrossfadeSamples_, false, false, true);
    }
    qualityTailBuffer_.clear();
    qualityTailValid_ = false;
    lastRealtimeMorphWasUnderrun_ = false;
}

int NeuralMorphingAudioProcessor::currentLatencySamplesForMode() const
{
    if (selectedProcessingMode() == ProcessingMode::QualityParity)
    {
        if (currentSampleRate_ <= 0.0)
            return qualitySuperframeSamples_;
        const int lookaheadSamples = static_cast<int>(
            std::ceil((static_cast<double>(qualityLookaheadMs_) / 1000.0) * currentSampleRate_));
        return juce::jmax(lookaheadSamples, qualitySuperframeSamples_);
    }

    return juce::jmax(juce::jmax(256, samplesPerBlock_ * 2), realtimeEncodeWindowSamples_);
}

void NeuralMorphingAudioProcessor::applyProcessingModeChangeIfNeeded()
{
    int modeParam = 0;
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("processingMode")))
        modeParam = param->getIndex();

    if (modeParam != lastKnownProcessingModeParam_)
    {
        lastKnownProcessingModeParam_ = modeParam;
        configureRealtimeTimings();
        currentLatencySamples_ = currentLatencySamplesForMode();
        setLatencySamples(currentLatencySamples_);
        morphUpdateCountdownSamples_ = 0;
        morphRefreshPending_.store(true, std::memory_order_release);
        realtimeWorkerWake_.signal();
    }
}

void NeuralMorphingAudioProcessor::clearRealtimeSessionState(bool clearHistoryBuffer)
{
    morphRevision_.fetch_add(1, std::memory_order_acq_rel);
    targetSegments_.clear();
    hasLastRealtimeMorphBlock_ = false;
    lastRealtimeMorphBlock_.setSize(0, 0);
    lastRealtimeMorphReadPosition_ = 0;
    lastRealtimeMorphWasUnderrun_ = false;
    morphUpdateCountdownSamples_ = 0;
    decodedFifo_.clear();
    if (clearHistoryBuffer)
    {
        realtimeInputHistory_.clear();
        realtimeInputFilledSamples_ = 0;
    }
    resetMorphSmoothing();
    lastMatchedIndex_.store(-1, std::memory_order_release);
    qualityTailBuffer_.clear();
    qualityTailValid_ = false;
    wetAvailabilityMix_ = 0.0f;
    wetMixPercent_.store(0, std::memory_order_release);
    morphWetState_.store(0, std::memory_order_release);
    morphTokenChangePercent_.store(-1, std::memory_order_release);
    morphAudioFingerprint_.store(0, std::memory_order_release);
    outputAudioFingerprint_.store(0, std::memory_order_release);
    clearLatestTargetPending_.store(true, std::memory_order_release);
    morphRefreshPending_.store(false, std::memory_order_release);

    const juce::ScopedLock taskLock(realtimeTaskMutex_);
    hasPendingRealtimeTask_ = false;
    pendingRealtimeEncodeInput_.clear();
}

NeuralMorphingAudioProcessor::ProcessingMode NeuralMorphingAudioProcessor::selectedProcessingMode() const
{
    int value = 0;
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("processingMode")))
        value = param->getIndex();
    return processingModeFromParam(value);
}

NeuralMorphingAudioProcessor::BackendPolicy NeuralMorphingAudioProcessor::selectedBackendPolicy() const
{
    int value = 0;
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("backend")))
        value = param->getIndex();
    return backendPolicyFromParam(value);
}

juce::String NeuralMorphingAudioProcessor::selectedCodecId() const
{
    int codecChoice = 0;
    if (auto* codecParam = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("bridgeCodec")))
        codecChoice = codecParam->getIndex();
    return (codecChoice == 1) ? "spectrostream" : "dac";
}

void NeuralMorphingAudioProcessor::updateWetAvailability(bool wetAvailable, int numSamples)
{
    if (currentSampleRate_ <= 0.0)
    {
        wetAvailabilityMix_ = wetAvailable ? 1.0f : 0.0f;
        wetMixPercent_.store(static_cast<int>(std::lround(getParam("dryWet") * wetAvailabilityMix_ * 100.0f)), std::memory_order_release);
        return;
    }

    if (wetAvailable)
    {
        wetAvailabilityMix_ = 1.0f;
        wetMixPercent_.store(static_cast<int>(std::lround(getParam("dryWet") * 100.0f)), std::memory_order_release);
        return;
    }

    const float fadeSeconds = 1.250f;
    const float fadeSamples = juce::jmax(1.0f, fadeSeconds * static_cast<float>(currentSampleRate_));
    const float step = juce::jlimit(0.0f, 1.0f, static_cast<float>(numSamples) / fadeSamples);
    const float target = 0.0f;
    if (target > wetAvailabilityMix_)
        wetAvailabilityMix_ = juce::jmin(1.0f, wetAvailabilityMix_ + step);
    else
        wetAvailabilityMix_ = juce::jmax(0.0f, wetAvailabilityMix_ - step);
    wetMixPercent_.store(static_cast<int>(std::lround(getParam("dryWet") * wetAvailabilityMix_ * 100.0f)), std::memory_order_release);
}

juce::String NeuralMorphingAudioProcessor::backendKindToString(ActiveBackendKind kind) const
{
    switch (kind)
    {
        case ActiveBackendKind::NativeOnnx:
            return "native";
        case ActiveBackendKind::BridgeHttp:
            return "py";
        case ActiveBackendKind::Stub:
        default:
            return "stub";
    }
}

juce::String NeuralMorphingAudioProcessor::processingModeToString(ProcessingMode mode) const
{
    return mode == ProcessingMode::QualityParity ? "quality" : "live";
}

juce::String NeuralMorphingAudioProcessor::backendPolicyToString(BackendPolicy policy) const
{
    switch (policy)
    {
        case BackendPolicy::BridgeOnly:
            return "bridge";
        case BackendPolicy::NativeOnly:
            return "native";
        case BackendPolicy::NativePreferredBridgeFallback:
        default:
            return "native+bridge";
    }
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
    visualWetLevel_.store(0.0f, std::memory_order_release);
    applyProcessingModeChangeIfNeeded();

    const int totalNumInputChannels = getTotalNumInputChannels();
    const int totalNumOutputChannels = getTotalNumOutputChannels();
    const int numSamples = buffer.getNumSamples();

    for (int channel = totalNumInputChannels; channel < totalNumOutputChannels; ++channel)
        buffer.clear(channel, 0, numSamples);

    if (resetSmoothingPending_.exchange(false, std::memory_order_acq_rel))
        resetMorphSmoothing();
    if (isStandaloneWrapper())
        renderStandaloneSource(buffer);

    if (dryScratch_.getNumChannels() != buffer.getNumChannels()
        || dryScratch_.getNumSamples() < numSamples)
    {
        dryScratch_.setSize(buffer.getNumChannels(), numSamples, false, false, true);
    }
    dryScratch_.makeCopyOf(buffer, true);
    const float dryWet = juce::jlimit(0.0f, 1.0f, getParam("dryWet"));

    // Hard bypass: if fully dry, avoid bridge/model work in the realtime callback.
    if (dryWet <= 0.0f)
    {
        morphWetState_.store(4, std::memory_order_release);

        const float outputGain = juce::Decibels::decibelsToGain(getParam("outputGain"));
        buffer.applyGain(outputGain);
        applySafetyLimiter(buffer);
        outputAudioFingerprint_.store(audioFingerprint(buffer), std::memory_order_release);
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
            morphWetState_.store(0, std::memory_order_release);
            updateWetAvailability(false, numSamples);
            buffer.makeCopyOf(dryScratch_, true);
            buffer.applyGain(1.0f - dryWet);
        }
        else
        {
            if (!isSilent(backendInputScratch_))
            {
                bool shouldQueueMorph = true;
                if (morphUpdateCountdownSamples_ > 0)
                {
                    morphUpdateCountdownSamples_ = juce::jmax(0, morphUpdateCountdownSamples_ - numSamples);
                    shouldQueueMorph = false;
                }
                else if (currentSampleRate_ > 0.0)
                {
                    morphUpdateCountdownSamples_ = juce::jmax(
                        1,
                        static_cast<int>(currentSampleRate_ * (static_cast<double>(morphUpdateIntervalMs_) / 1000.0)));
                }

                if (shouldQueueMorph)
                {
                    // Avoid querying while the palette is actively rebuilding.
                    if (paletteWorker_ == nullptr || !paletteWorker_->isBusy())
                    {
                        const auto& encodeInput = selectRealtimeEncodeInput(backendInputScratch_);
                        queueRealtimeMorphTask(encodeInput);
                        if (!hasLastRealtimeMorphBlock_)
                            morphWetState_.store(1, std::memory_order_release);
                    }
                }

                juce::AudioBuffer<float> latestMorphed;
                while (decodedFifo_.pop(latestMorphed))
                {
                    if (latestMorphed.getNumChannels() <= 0 || latestMorphed.getNumSamples() <= 0)
                        continue;
                    wetTransitionPending_ = hasLastRealtimeMorphBlock_;
                    lastRealtimeMorphBlock_.makeCopyOf(latestMorphed);
                    hasLastRealtimeMorphBlock_ = true;
                    lastRealtimeMorphReadPosition_ = 0;
                    lastRealtimeMorphWasUnderrun_ = false;
                }

                if (hasLastRealtimeMorphBlock_ && lastRealtimeMorphBlock_.getNumSamples() > 0)
                {
                    const bool enoughWetSamples = lastRealtimeMorphReadPosition_ + numSamples <= lastRealtimeMorphBlock_.getNumSamples();
                    if (!enoughWetSamples)
                    {
                        lastRealtimeMorphWasUnderrun_ = true;
                        wetTransitionPending_ = true;
                        lastRealtimeMorphReadPosition_ = 0;
                    }
                    morphWetState_.store(lastRealtimeMorphWasUnderrun_ ? 3 : 2, std::memory_order_release);
                    updateWetAvailability(true, numSamples);
                    mixMorphedAudio(buffer, dryScratch_, lastRealtimeMorphBlock_);
                }
                else
                {
                    morphWetState_.store(1, std::memory_order_release);
                    updateWetAvailability(false, numSamples);
                    buffer.makeCopyOf(dryScratch_, true);
                    buffer.applyGain(1.0f - dryWet);
                }
            }
            else
            {
                clearRealtimeSessionState(true);
                morphWetState_.store(4, std::memory_order_release);
                buffer.makeCopyOf(dryScratch_, true);
                buffer.applyGain(1.0f - dryWet);
            }
        }
    }
    else
    {
        morphWetState_.store(0, std::memory_order_release);
        updateWetAvailability(false, numSamples);
        buffer.makeCopyOf(dryScratch_, true);
        buffer.applyGain(1.0f - dryWet);
    }

    const float outputGain = juce::Decibels::decibelsToGain(getParam("outputGain"));
    buffer.applyGain(outputGain);
    applySafetyLimiter(buffer);
    outputAudioFingerprint_.store(audioFingerprint(buffer), std::memory_order_release);
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

    applyDacDemoDefaults();
    const int backendIndex = static_cast<int>(getParam("backend"));
    switchBackend(backendIndex);
    configureRealtimeTimings();
    currentLatencySamples_ = currentLatencySamplesForMode();
    setLatencySamples(currentLatencySamples_);
}

float NeuralMorphingAudioProcessor::getParam(const juce::String& paramID) const
{
    if (auto* value = parameters.getRawParameterValue(paramID))
        return *value;
    return 0.0f;
}

void NeuralMorphingAudioProcessor::parameterChanged(const juce::String& parameterID, float)
{
    morphRevision_.fetch_add(1, std::memory_order_acq_rel);
    morphRefreshPending_.store(true, std::memory_order_release);
    lastMatchedIndex_.store(-1, std::memory_order_release);

    if ((parameterID == "unit" || parameterID == "stride") && paletteWorker_ != nullptr)
    {
        const int unit = juce::jmax(1, static_cast<int>(getParam("unit")));
        const int stride = juce::jlimit(1, unit, static_cast<int>(getParam("stride")));
        paletteWorker_->requestRegrain(unit, stride);
    }

    realtimeWorkerWake_.signal();
}

void NeuralMorphingAudioProcessor::setStandaloneSource(juce::AudioBuffer<float> buffer, double sampleRate, const juce::String& name)
{
    {
        const juce::ScopedLock lock(standaloneMutex_);
        standaloneSourceBuffer_ = std::move(buffer);
        standaloneSourceSampleRate_ = sampleRate;
        standaloneSourcePosition_ = 0;
        standaloneSourceName_ = name;
        standaloneSourceLoaded_ = standaloneSourceBuffer_.getNumSamples() > 0;
    }
    clearRealtimeSessionState(true);
}

void NeuralMorphingAudioProcessor::clearStandaloneSource()
{
    {
        const juce::ScopedLock lock(standaloneMutex_);
        standaloneSourceBuffer_.setSize(0, 0);
        standaloneSourceSampleRate_ = 0.0;
        standaloneSourcePosition_ = 0;
        standaloneSourceName_.clear();
        standaloneSourceLoaded_ = false;
    }
    clearRealtimeSessionState(true);
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

void NeuralMorphingAudioProcessor::invalidateMorphCache(bool keepWet)
{
    {
        const juce::SpinLock::ScopedLockType lock(morphCacheMutex_);
        morphCache_.clear();
    }

    morphRevision_.fetch_add(1, std::memory_order_acq_rel);
    lastMatchedIndex_.store(-1, std::memory_order_release);
    if (keepWet)
    {
        morphRefreshPending_.store(true, std::memory_order_release);
        realtimeWorkerWake_.signal();
    }
    else
    {
        clearRealtimeSessionState(true);
        resetSmoothingPending_.store(true, std::memory_order_release);
    }
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

    if (deterministicDemoPlayback_ && !paletteReady())
    {
        buffer.clear();
        return true;
    }

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
    const auto channels = static_cast<size_t>(juce::jmax(1, getTotalNumOutputChannels()));
    sourceEnvelopeState_.assign(channels, 0.0f);
    wetEnvelopeState_.assign(channels, 0.0f);
    lastWetOutputSample_.assign(channels, 0.0f);
    wetTransitionFromSample_.assign(channels, 0.0f);
    wetTransitionRemainingSamples_ = 0;
    wetTransitionTotalSamples_ = 0;
    wetTransitionPending_ = false;
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

uint64_t NeuralMorphingAudioProcessor::hashMorphKey(const TokenBlock& block, double outputSampleRate) const
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
    mix(static_cast<uint64_t>(std::lround(getParam("similarity") * 1000.0f)));
    mix(static_cast<uint64_t>(std::lround(getParam("swapMode"))));
    mix(static_cast<uint64_t>(std::lround(getParam("processingMode"))));
    mix(static_cast<uint64_t>(std::lround(outputSampleRate)));

    return hash;
}

juce::AudioProcessorValueTreeState::ParameterLayout NeuralMorphingAudioProcessor::createParameterLayout()
{
    using R = juce::NormalisableRange<float>;
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>("temperature", "Temperature", R(0.1f, 2.0f, 0.01f), 0.47f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("threshold", "Threshold", R(0.1f, 2.0f, 0.01f), 0.99f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("continuity", "Continuity", R(0.0f, 1.0f, 0.01f), 0.10f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("rvqFocus", "RVQ Focus", R(0.0f, 1.0f, 0.01f), 0.30f));
    params.push_back(std::make_unique<juce::AudioParameterInt>("unit", "Grain Size", 1, 10, 7));
    params.push_back(std::make_unique<juce::AudioParameterInt>("stride", "Grain Step", 1, 10, 2));
    juce::StringArray swapModeChoices;
    swapModeChoices.add("Full Layer");
    swapModeChoices.add("RVQ Group");
    swapModeChoices.add("Palette Only");
    params.push_back(std::make_unique<juce::AudioParameterChoice>("swapMode", "Swap Mode", swapModeChoices, 2));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("similarity", "Palette Bias", R(0.0f, 1.0f, 0.01f), 1.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("envelopeFollow", "Envelope", R(0.0f, 1.0f, 0.01f), 0.05f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("dryWet", "Dry/Wet", R(0.0f, 1.0f, 0.01f), 0.9f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("outputGain", "Output Gain (dB)", R(-24.0f, 24.0f, 0.1f), -3.0f));

    juce::StringArray processingModeChoices;
    processingModeChoices.add("Quality Parity");
    processingModeChoices.add("Live Realtime");
    params.push_back(std::make_unique<juce::AudioParameterChoice>("processingMode", "Processing Mode", processingModeChoices, 0));

    // Backend policy (keeps parameter id for backward-compatible state loading)
    // 0 = native preferred + bridge fallback, 1 = bridge only, 2 = native only
    juce::StringArray backendChoices;
    backendChoices.add("Native Preferred (+ Bridge Fallback)");
    backendChoices.add("Bridge Only");
    backendChoices.add("Native Only");
    params.push_back(std::make_unique<juce::AudioParameterChoice>("backend", "Backend", backendChoices, 0));

    // Bridge codec selection (used by HTTP backend, ignored by native backends)
    juce::StringArray bridgeCodecChoices;
    bridgeCodecChoices.add("DAC");
    bridgeCodecChoices.add("SpectroStream");
    params.push_back(std::make_unique<juce::AudioParameterChoice>("bridgeCodec", "Bridge Codec", bridgeCodecChoices, 0));

    return { params.begin(), params.end() };
}

void NeuralMorphingAudioProcessor::applySafetyLimiter(juce::AudioBuffer<float>& buffer)
{
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    if (numChannels <= 0 || numSamples <= 0)
        return;

    float peak = 0.0f;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const auto* data = buffer.getReadPointer(ch);
        for (int i = 0; i < numSamples; ++i)
            peak = std::max(peak, std::abs(data[i]));
    }

    if (peak < 1.0e-5f)
    {
        outputSafetyGain_ = 1.0f;
        return;
    }

    constexpr float targetPeak = 0.8912509f; // -1 dBFS
    const float desiredGain = (peak > targetPeak && peak > 0.0f) ? (targetPeak / peak) : 1.0f;
    const float attack = 0.45f;
    const float release = 0.08f;
    const float coeff = (desiredGain < outputSafetyGain_) ? attack : release;
    outputSafetyGain_ += coeff * (desiredGain - outputSafetyGain_);
    outputSafetyGain_ = juce::jlimit(0.10f, 1.0f, outputSafetyGain_);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            const float limited = data[i] * outputSafetyGain_;
            data[i] = juce::jlimit(-0.95f, 0.95f, limited);
        }
    }
}

TokenBlock NeuralMorphingAudioProcessor::buildMatchedTokenBlock(const TokenBlock& targetBlock)
{
    TokenBlock result;

    if (!paletteReady() || backend_ == nullptr)
        return result;

    const auto paletteSnapshot = paletteIndex_->snapshot();
    if (paletteSnapshot == nullptr || paletteSnapshot->sources == nullptr || paletteSnapshot->size() <= 0)
        return result;

    if (targetBlock.tokens.empty() || targetBlock.frames <= 0)
        return result;

    result = targetBlock;

    const int unit = juce::jmax(1, static_cast<int>(getParam("unit")));
    const int stride = juce::jlimit(1, unit, static_cast<int>(getParam("stride")));
    if (paletteSnapshot->grainConfig.unit != unit || paletteSnapshot->grainConfig.stride != stride)
        return {};
    const int swapMode = juce::jlimit(0, 2, static_cast<int>(std::lround(getParam("swapMode"))));
    const bool usePaletteOnlySwap = (swapMode == 2);
    const bool useFullLayerSwap = (swapMode == 0 || usePaletteOnlySwap);
    if (usePaletteOnlySwap)
        std::fill(result.tokens.begin(), result.tokens.end(), 0);

    if (targetBlock.frames < unit)
    {
        if (usePaletteOnlySwap && !paletteSnapshot->metas.empty())
        {
            const auto& meta = paletteSnapshot->metas.front();
            if (meta.sampleId >= 0
                && static_cast<size_t>(meta.sampleId) < paletteSnapshot->sources->tokenBlocks.size())
            {
                const auto& paletteBlock = paletteSnapshot->sources->tokenBlocks[static_cast<size_t>(meta.sampleId)];
                if (paletteBlock.empty())
                    return result;
                for (int codebook = 0; codebook < result.codebooks; ++codebook)
                {
                    for (int frame = 0; frame < result.frames; ++frame)
                    {
                        const int paletteFrame = juce::jlimit(0, paletteBlock.frames - 1, meta.frame + frame);
                        const int sourceIndex = paletteBlock.index(codebook, paletteFrame);
                        if (sourceIndex >= 0 && static_cast<size_t>(sourceIndex) < paletteBlock.tokens.size())
                            result.tokens[static_cast<size_t>(result.index(codebook, frame))] = paletteBlock.tokens[static_cast<size_t>(sourceIndex)];
                    }
                }
            }
        }
        return result;
    }

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
    const float continuityWeight = 4.0f * continuity * continuity;
    const float paletteBias = juce::jlimit(0.0f, 1.0f, getParam("similarity"));
    const float temperatureNorm = juce::jlimit(0.0f, 1.0f, (temperature - 0.1f) / 1.9f);
    const uint64_t targetHash = hashTokenBlock(targetBlock);

    const bool qualityMode = selectedProcessingMode() == ProcessingMode::QualityParity;
    const int candidateBudget = qualityMode ? matchCandidateCount : liveCandidateCount_;
    const int beamBudget = qualityMode ? matchBeamWidth : liveBeamWidth_;

    const int candidateCount = std::min(candidateBudget, paletteSnapshot->size());
    if (candidateCount <= 0)
        return result;

    const int beamWidth = std::min(beamBudget, candidateCount);

    struct CandidateInfo
    {
        int annIndex = -1;
        float emission = 0.0f;
        float fineDistance = 0.0f;
        float selectionCost = 0.0f;
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
        std::vector<float> fallbackRow;
        for (int offset = 0; offset < unit; ++offset)
        {
            const std::vector<float>* row = nullptr;
            if (hasFrameVectors)
            {
                row = &frameVectors[static_cast<size_t>(startFrame + offset)];
            }
            else
            {
                fallbackRow = backend_->tokensToVectorRow(targetBlock, startFrame + offset);
                row = &fallbackRow;
            }
            if (static_cast<int>(row->size()) != embeddingDim)
                return false;

            for (int d = 0; d < embeddingDim; ++d)
                out[static_cast<size_t>(d)] += (*row)[static_cast<size_t>(d)];
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

        RvqSearchConfig search;
        search.coarseLength = coarseLen;
        search.midLength = midLen;
        search.fineLength = fineLen;
        search.coarseWeight = weightCoarse;
        search.midWeight = weightMid;
        search.fineWeight = weightFine;
        auto matches = paletteIndex_->query(paletteSnapshot, queryVector, candidateCount, search);
        if (matches.empty())
            continue;

        GrainMatch grain;
        grain.startFrame = startFrame;
        grain.candidates.reserve(matches.size());

        for (const auto& match : matches)
        {
            if (match.distance <= threshold)
                grain.candidates.push_back({ match.annIndex, match.distance, match.fineDistance, 0.0f });
        }

        if (grain.candidates.empty())
        {
            const auto& nearest = matches.front();
            grain.candidates.push_back({ nearest.annIndex, nearest.distance, nearest.fineDistance, 0.0f });
        }

        if (grain.candidates.empty())
            continue;

        std::sort(grain.candidates.begin(), grain.candidates.end(),
                  [](const CandidateInfo& a, const CandidateInfo& b)
                  {
                      if (a.emission == b.emission)
                          return a.annIndex < b.annIndex;
                      return a.emission < b.emission;
                  });

        const int lastRank = static_cast<int>(grain.candidates.size()) - 1;
        const int preferredRank = juce::jlimit(
            0,
            lastRank,
            static_cast<int>(std::lround((1.0f - paletteBias) * static_cast<float>(lastRank))));
        const int radius = static_cast<int>(std::lround(temperatureNorm * static_cast<float>(lastRank)));
        int sampledRank = preferredRank;
        if (radius > 0)
        {
            const uint32_t seed = static_cast<uint32_t>(targetHash)
                                  ^ static_cast<uint32_t>(targetHash >> 32U)
                                  ^ static_cast<uint32_t>(startFrame * 2654435761U);
            std::mt19937 random(seed);
            std::uniform_int_distribution<int> offset(-radius, radius);
            sampledRank = juce::jlimit(0, lastRank, preferredRank + offset(random));
        }

        const float rankScale = 1.0f / static_cast<float>(juce::jmax(1, lastRank));
        for (int rank = 0; rank <= lastRank; ++rank)
        {
            auto& candidate = grain.candidates[static_cast<size_t>(rank)];
            candidate.selectionCost = std::abs(static_cast<float>(rank - sampledRank)) * rankScale
                                      + 0.05f * candidate.emission;
        }

        grains.push_back(std::move(grain));
    }

    if (grains.empty())
        return result;

    auto transitionCost = [&](int prevIndex, int nextIndex) -> float
    {
        const auto& prevMeta = paletteSnapshot->metas.at(static_cast<size_t>(prevIndex));
        const auto& nextMeta = paletteSnapshot->metas.at(static_cast<size_t>(nextIndex));
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

        const float latentPenalty = paletteIndex_->cosineDistance(paletteSnapshot, prevIndex, nextIndex);
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
        float score = candidate.selectionCost;
        const int previousMatch = lastMatchedIndex_.load(std::memory_order_acquire);
        if (continuityWeight > 0.0f && previousMatch >= 0 && previousMatch < paletteSnapshot->size())
            score += continuityWeight * transitionCost(previousMatch, candidate.annIndex);
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
                float score = prevState.score + candidate.selectionCost;
                if (continuityWeight > 0.0f)
                    score += continuityWeight * transitionCost(prevAnn, candidate.annIndex);
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
        int candidateIdx = path[g];
        if (candidateIdx < 0 || static_cast<size_t>(candidateIdx) >= grain.candidates.size())
            continue;
        const auto& best = grain.candidates[static_cast<size_t>(candidateIdx)];
        lastMatchedIndex_.store(best.annIndex, std::memory_order_release);
        const bool fallbackCoarse = best.emission > threshold;

        const int startFrame = grain.startFrame;
        const int requestedSpan = (usePaletteOnlySwap && g + 1 == grains.size()) ? (result.frames - startFrame) : unit;
        const int span = juce::jmin(requestedSpan, result.frames - startFrame);
        if (span <= 0)
            continue;
        const auto& bestMeta = paletteSnapshot->metas.at(static_cast<size_t>(best.annIndex));
        const TokenBlock* bestBlock = nullptr;
        if (bestMeta.sampleId >= 0
            && static_cast<size_t>(bestMeta.sampleId) < paletteSnapshot->sources->tokenBlocks.size())
        {
            bestBlock = &paletteSnapshot->sources->tokenBlocks[static_cast<size_t>(bestMeta.sampleId)];
        }

        if (useFullLayerSwap)
        {
            for (int q = 0; q < codebooks; ++q)
            {
                for (int offset = 0; offset < span; ++offset)
                {
                    int tokenValue = 0;
                    if (bestBlock != nullptr && bestBlock->frames > 0)
                    {
                        const int sourceFrame = juce::jmin(bestMeta.frame + offset, bestBlock->frames - 1);
                        const int srcIdx = bestBlock->index(q, sourceFrame);
                        if (srcIdx >= 0 && static_cast<size_t>(srcIdx) < bestBlock->tokens.size())
                            tokenValue = bestBlock->tokens[static_cast<size_t>(srcIdx)];
                    }
                    else if (!usePaletteOnlySwap)
                    {
                        const int fallbackIdx = targetBlock.index(q, startFrame + offset);
                        if (fallbackIdx >= 0 && static_cast<size_t>(fallbackIdx) < targetBlock.tokens.size())
                            tokenValue = targetBlock.tokens[static_cast<size_t>(fallbackIdx)];
                    }

                    result.tokens[static_cast<size_t>(result.index(q, startFrame + offset))] = tokenValue;
                }
            }
            continue;
        }

        const int topKMixCount = selectedCodecId() == "spectrostream" ? topKMixCountSpectroStream : topKMixCountDac;
        const int kCount = std::min(topKMixCount, static_cast<int>(grain.candidates.size()));
        std::vector<int> topIndices;
        std::vector<float> topDistances;
        topIndices.reserve(static_cast<size_t>(kCount));
        topDistances.reserve(static_cast<size_t>(kCount));

        const int firstTopIndex = juce::jlimit(
            0,
            static_cast<int>(grain.candidates.size()) - kCount,
            candidateIdx - kCount / 2);
        for (int i = 0; i < kCount; ++i)
        {
            const auto& topCandidate = grain.candidates[static_cast<size_t>(firstTopIndex + i)];
            topIndices.push_back(topCandidate.annIndex);
            topDistances.push_back(topCandidate.fineDistance);
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
                        const auto& meta = paletteSnapshot->metas.at(static_cast<size_t>(topIndices[static_cast<size_t>(i)]));
                        const TokenBlock* block = nullptr;
                        if (meta.sampleId >= 0
                            && static_cast<size_t>(meta.sampleId) < paletteSnapshot->sources->tokenBlocks.size())
                        {
                            block = &paletteSnapshot->sources->tokenBlocks[static_cast<size_t>(meta.sampleId)];
                        }
                        int tokenValue = 0;
                        if (block != nullptr && block->frames > 0)
                        {
                            const int sourceFrame = juce::jmin(meta.frame + offset, block->frames - 1);
                            const int idx = block->index(q, sourceFrame);
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
                if (bestBlock != nullptr && bestBlock->frames > 0)
                {
                    const int sourceFrame = juce::jmin(bestMeta.frame + offset, bestBlock->frames - 1);
                    const int srcIdx = bestBlock->index(q, sourceFrame);
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

    const float dryWet = juce::jlimit(0.0f, 1.0f, getParam("dryWet"));
    const float dryGain = 1.0f - dryWet;
    const float wetGain = dryWet;
    const float envelopeAmount = juce::jlimit(0.0f, 1.0f, getParam("envelopeFollow"));
    const bool followEnvelope = envelopeAmount > 0.0f && currentSampleRate_ > 0.0;

    if (!followEnvelope
        || sourceEnvelopeState_.size() < static_cast<size_t>(totalNumOutputChannels)
        || wetEnvelopeState_.size() < static_cast<size_t>(totalNumOutputChannels))
    {
        resetMorphSmoothing();
    }

    const int morphSamples = morphed.getNumSamples();
    const int readStart = juce::jlimit(0, juce::jmax(0, morphSamples), lastRealtimeMorphReadPosition_);
    if (wetTransitionPending_)
    {
        std::copy(lastWetOutputSample_.begin(), lastWetOutputSample_.end(), wetTransitionFromSample_.begin());
        wetTransitionTotalSamples_ = juce::jlimit(
            16,
            128,
            static_cast<int>(std::lround(0.002 * juce::jmax(1.0, currentSampleRate_))));
        wetTransitionRemainingSamples_ = wetTransitionTotalSamples_;
        wetTransitionPending_ = false;
    }
    const int transitionRemainingAtStart = wetTransitionRemainingSamples_;
    double wetEnergy = 0.0;
    int wetSampleCount = 0;
    const float attack = followEnvelope
                             ? std::exp(-1.0f / (0.005f * static_cast<float>(currentSampleRate_)))
                             : 0.0f;
    const float release = followEnvelope
                              ? std::exp(-1.0f / (0.080f * static_cast<float>(currentSampleRate_)))
                              : 0.0f;

    for (int ch = 0; ch < totalNumOutputChannels; ++ch)
    {
        const int morphCh = juce::jmin(ch, morphed.getNumChannels() - 1);
        const int dryCh = juce::jmin(ch, dryBuffer.getNumChannels() - 1);
        float sourceEnvelope = sourceEnvelopeState_[static_cast<size_t>(ch)];
        float wetEnvelope = wetEnvelopeState_[static_cast<size_t>(ch)];

        for (int sample = 0; sample < numSamples; ++sample)
        {
            const float drySample = dryBuffer.getSample(dryCh, sample);
            const int morphLinearIndex = readStart + sample;
            const bool hasMorphSample = morphLinearIndex < morphSamples;
            float morphSample = hasMorphSample ? morphed.getSample(morphCh, morphLinearIndex) : 0.0f;

            if (sample < transitionRemainingAtStart)
            {
                const int progress = wetTransitionTotalSamples_ - transitionRemainingAtStart + sample + 1;
                const float alpha = static_cast<float>(progress)
                                    / static_cast<float>(wetTransitionTotalSamples_ + 1);
                morphSample = wetTransitionFromSample_[static_cast<size_t>(ch)] * (1.0f - alpha)
                              + morphSample * alpha;
            }

            if (followEnvelope)
            {
                const float sourceMagnitude = std::abs(drySample);
                const float wetMagnitude = std::abs(morphSample);
                const float sourceCoefficient = sourceMagnitude > sourceEnvelope ? attack : release;
                const float wetCoefficient = wetMagnitude > wetEnvelope ? attack : release;
                sourceEnvelope = sourceCoefficient * sourceEnvelope + (1.0f - sourceCoefficient) * sourceMagnitude;
                wetEnvelope = wetCoefficient * wetEnvelope + (1.0f - wetCoefficient) * wetMagnitude;

                const float targetGain = juce::jlimit(0.0f, 4.0f, sourceEnvelope / (wetEnvelope + 1.0e-4f));
                morphSample *= 1.0f + envelopeAmount * (targetGain - 1.0f);
            }

            wetEnergy += static_cast<double>(morphSample) * static_cast<double>(morphSample);
            ++wetSampleCount;
            lastWetOutputSample_[static_cast<size_t>(ch)] = morphSample;

            buffer.setSample(ch, sample, dryGain * drySample + wetGain * morphSample);
        }

        if (followEnvelope)
        {
            sourceEnvelopeState_[static_cast<size_t>(ch)] = sourceEnvelope;
            wetEnvelopeState_[static_cast<size_t>(ch)] = wetEnvelope;
        }
    }

    if (morphSamples > 0)
        lastRealtimeMorphReadPosition_ = juce::jmin(readStart + numSamples, morphSamples);
    wetTransitionRemainingSamples_ = juce::jmax(0, transitionRemainingAtStart - numSamples);

    if (wetSampleCount > 0)
        visualWetLevel_.store(static_cast<float>(std::sqrt(wetEnergy / static_cast<double>(wetSampleCount))),
                              std::memory_order_release);
}

bool NeuralMorphingAudioProcessor::paletteReady() const
{
    return paletteIndex_ != nullptr && paletteIndex_->size() > 0;
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
    stopRealtimeWorker();

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

    startRealtimeWorker();
}

void NeuralMorphingAudioProcessor::startRealtimeWorker()
{
    if (realtimeWorkerThread_.joinable())
        return;

    realtimeWorkerShouldExit_.store(false, std::memory_order_release);
    realtimeWorkerThread_ = std::thread([this] { realtimeWorkerLoop(); });
}

void NeuralMorphingAudioProcessor::stopRealtimeWorker()
{
    realtimeWorkerShouldExit_.store(true, std::memory_order_release);
    realtimeWorkerWake_.signal();
    if (realtimeWorkerThread_.joinable())
        realtimeWorkerThread_.join();

    const juce::ScopedLock taskLock(realtimeTaskMutex_);
    hasPendingRealtimeTask_ = false;
    pendingRealtimeEncodeInput_.clear();
    latestTargetTokens_ = {};
    latestTargetInput_.setSize(0, 0);
}

void NeuralMorphingAudioProcessor::queueRealtimeMorphTask(const juce::AudioBuffer<float>& encodeInput)
{
    if (realtimeWorkerShouldExit_.load(std::memory_order_acquire))
        return;

    const juce::CriticalSection::ScopedTryLockType taskLock(realtimeTaskMutex_);
    if (!taskLock.isLocked())
        return;

    if (pendingRealtimeEncodeInput_.getNumChannels() != encodeInput.getNumChannels()
        || pendingRealtimeEncodeInput_.getNumSamples() != encodeInput.getNumSamples())
    {
        pendingRealtimeEncodeInput_.setSize(encodeInput.getNumChannels(), encodeInput.getNumSamples(), false, false, true);
    }
    pendingRealtimeEncodeInput_.makeCopyOf(encodeInput, true);
    pendingRealtimeSampleRate_ = currentSampleRate_;
    hasPendingRealtimeTask_ = true;
    realtimeWorkerWake_.signal();
}

void NeuralMorphingAudioProcessor::realtimeWorkerLoop()
{
    while (!realtimeWorkerShouldExit_.load(std::memory_order_acquire))
    {
        realtimeWorkerWake_.wait(100);
        if (realtimeWorkerShouldExit_.load(std::memory_order_acquire))
            break;

        if (clearLatestTargetPending_.exchange(false, std::memory_order_acq_rel))
        {
            latestTargetTokens_ = {};
            latestTargetInput_.setSize(0, 0);
        }

        const bool refreshRequested = morphRefreshPending_.exchange(false, std::memory_order_acq_rel);
        juce::AudioBuffer<float> encodeInput;
        double inputSampleRate = currentSampleRate_;
        bool hasInput = false;
        {
            const juce::ScopedLock taskLock(realtimeTaskMutex_);
            if (hasPendingRealtimeTask_)
            {
                encodeInput.makeCopyOf(pendingRealtimeEncodeInput_, true);
                inputSampleRate = pendingRealtimeSampleRate_;
                hasPendingRealtimeTask_ = false;
                hasInput = true;
            }
        }

        if (!hasInput && !refreshRequested)
            continue;

        if (paletteWorker_ != nullptr && paletteWorker_->isBusy())
        {
            if (refreshRequested)
                morphRefreshPending_.store(true, std::memory_order_release);
            continue;
        }

        if (backend_ == nullptr || !backend_->ready() || !paletteReady())
            continue;

        if (hasInput)
        {
            latestTargetTokens_ = backend_->encodePCM(encodeInput, inputSampleRate);
            if (latestTargetTokens_.empty())
                continue;
            latestTargetInput_.makeCopyOf(encodeInput, true);
            latestTargetOutputSampleRate_ = inputSampleRate;
        }

        if (latestTargetTokens_.empty())
            continue;

        const uint64_t revision = morphRevision_.load(std::memory_order_acquire);
        processRealtimeMorphTokens(
            latestTargetTokens_,
            latestTargetInput_,
            latestTargetOutputSampleRate_,
            revision);
    }
}

void NeuralMorphingAudioProcessor::processRealtimeMorphTokens(const TokenBlock& targetTokens,
                                                              const juce::AudioBuffer<float>& sourceAudio,
                                                              double outputSampleRate,
                                                              uint64_t revision)
{
    juce::ignoreUnused(sourceAudio);

    juce::AudioBuffer<float> morphedAudio;
    int tokenChange = -1;
    juce::String error;
    if (!renderMorphedTokenBlock(
            targetTokens,
            outputSampleRate,
            morphedAudio,
            tokenChange,
            error,
            revision))
    {
        if (revision != morphRevision_.load(std::memory_order_acquire))
        {
            lastMatchedIndex_.store(-1, std::memory_order_release);
            morphRefreshPending_.store(true, std::memory_order_release);
            realtimeWorkerWake_.signal();
        }
        return;
    }

    matchWetLoudness(sourceAudio, morphedAudio);

    if (revision != morphRevision_.load(std::memory_order_acquire))
    {
        lastMatchedIndex_.store(-1, std::memory_order_release);
        morphRefreshPending_.store(true, std::memory_order_release);
        realtimeWorkerWake_.signal();
        return;
    }

    morphTokenChangePercent_.store(tokenChange, std::memory_order_release);

    const int requestedCrossfadeSamples = selectedProcessingMode() == ProcessingMode::QualityParity
                                              ? qualityCrossfadeSamples_
                                              : static_cast<int>(std::lround(0.010 * outputSampleRate));
    if (requestedCrossfadeSamples > 0)
    {
        const int crossfadeSamples = juce::jmin(
            requestedCrossfadeSamples,
            juce::jmin(morphedAudio.getNumSamples(), qualityTailBuffer_.getNumSamples()));

        if (qualityTailValid_ && crossfadeSamples > 0 && qualityTailBuffer_.getNumChannels() > 0)
        {
            for (int ch = 0; ch < morphedAudio.getNumChannels(); ++ch)
            {
                const int tailCh = juce::jmin(ch, qualityTailBuffer_.getNumChannels() - 1);
                for (int i = 0; i < crossfadeSamples; ++i)
                {
                    const float alpha = static_cast<float>(i + 1) / static_cast<float>(crossfadeSamples + 1);
                    const float previous = qualityTailBuffer_.getSample(tailCh, i);
                    const float current = morphedAudio.getSample(ch, i);
                    const float blended = (1.0f - alpha) * previous + alpha * current;
                    morphedAudio.setSample(ch, i, blended);
                }
            }
        }

        if (crossfadeSamples > 0)
        {
            if (qualityTailBuffer_.getNumChannels() != morphedAudio.getNumChannels())
                qualityTailBuffer_.setSize(morphedAudio.getNumChannels(), qualityCrossfadeSamples_, false, false, true);
            qualityTailBuffer_.clear();

            const int tailStart = juce::jmax(0, morphedAudio.getNumSamples() - crossfadeSamples);
            for (int ch = 0; ch < qualityTailBuffer_.getNumChannels(); ++ch)
                qualityTailBuffer_.copyFrom(ch, 0, morphedAudio, ch, tailStart, crossfadeSamples);
            qualityTailValid_ = true;
        }
        else
        {
            qualityTailValid_ = false;
        }
    }
    else
    {
        qualityTailValid_ = false;
    }

    if (revision != morphRevision_.load(std::memory_order_acquire))
    {
        lastMatchedIndex_.store(-1, std::memory_order_release);
        morphRefreshPending_.store(true, std::memory_order_release);
        realtimeWorkerWake_.signal();
        return;
    }

    morphAudioFingerprint_.store(audioFingerprint(morphedAudio), std::memory_order_release);

    if (!decodedFifo_.push(std::move(morphedAudio)))
    {
        juce::AudioBuffer<float> dropped;
        decodedFifo_.pop(dropped);
        decodedFifo_.push(std::move(morphedAudio));
    }
}

bool NeuralMorphingAudioProcessor::renderMorphedAudioForInput(const juce::AudioBuffer<float>& encodeInput,
                                                              double inputSampleRate,
                                                              double outputSampleRate,
                                                              juce::AudioBuffer<float>& morphedAudio,
                                                              int& tokenChangePercentOut,
                                                              juce::String& error,
                                                              uint64_t expectedRevision)
{
    try
    {
        if (backend_ == nullptr || !backend_->ready())
        {
            error = "Backend is not ready.";
            return false;
        }

        auto targetTokens = backend_->encodePCM(encodeInput, inputSampleRate);
        if (targetTokens.empty())
        {
            error = "Encode produced no tokens.";
            return false;
        }

        const bool rendered = renderMorphedTokenBlock(
            targetTokens,
            outputSampleRate,
            morphedAudio,
            tokenChangePercentOut,
            error,
            expectedRevision);
        if (rendered)
            matchWetLoudness(encodeInput, morphedAudio);
        return rendered;
    }
    catch (const std::exception& ex)
    {
        error = ex.what();
        return false;
    }
    catch (...)
    {
        error = "Unknown encode error.";
        return false;
    }
}

bool NeuralMorphingAudioProcessor::renderMorphedTokenBlock(const TokenBlock& targetTokens,
                                                           double outputSampleRate,
                                                           juce::AudioBuffer<float>& morphedAudio,
                                                           int& tokenChangePercentOut,
                                                           juce::String& error,
                                                           uint64_t expectedRevision)
{
    error.clear();
    tokenChangePercentOut = -1;
    morphedAudio.setSize(0, 0);

    if (backend_ == nullptr || !backend_->ready())
    {
        error = "Backend is not ready.";
        return false;
    }

    if (!paletteReady())
    {
        error = "Palette is not ready.";
        return false;
    }

    if (expectedRevision != 0 && expectedRevision != morphRevision_.load(std::memory_order_acquire))
    {
        error = "Morph request superseded.";
        return false;
    }

    try
    {
        const uint64_t cacheKey = hashMorphKey(targetTokens, outputSampleRate);
        {
            const juce::SpinLock::ScopedLockType lock(morphCacheMutex_);
            for (size_t i = 0; i < morphCache_.size(); ++i)
            {
                if (morphCache_[i].hash != cacheKey)
                    continue;

                if (i > 0)
                {
                    auto entry = std::move(morphCache_[i]);
                    morphCache_.erase(morphCache_.begin() + static_cast<long>(i));
                    morphCache_.insert(morphCache_.begin(), std::move(entry));
                }

                if (morphCache_.front().audio.getNumSamples() > 0)
                {
                    if (expectedRevision != 0
                        && expectedRevision != morphRevision_.load(std::memory_order_acquire))
                    {
                        error = "Morph request superseded.";
                        return false;
                    }
                    morphedAudio.makeCopyOf(morphCache_.front().audio);
                    tokenChangePercentOut = morphCache_.front().tokenChangePercent;
                    return true;
                }
                break;
            }
        }

        if (paletteWorker_ != nullptr && paletteWorker_->isBusy())
        {
            error = "Palette is still rebuilding.";
            return false;
        }

        auto matchedTokens = buildMatchedTokenBlock(targetTokens);
        if (matchedTokens.tokens.empty() || matchedTokens.frames <= 0)
        {
            error = "Morph matching produced no tokens.";
            return false;
        }

        tokenChangePercentOut = tokenChangePercent(targetTokens, matchedTokens);
        morphedAudio = backend_->decodeTokens(matchedTokens);
        if (morphedAudio.getNumChannels() <= 0 || morphedAudio.getNumSamples() <= 0)
        {
            error = "Decode produced no audio.";
            return false;
        }

        morphedAudio = resampleAudio(
            morphedAudio,
            static_cast<double>(backend_->sampleRate()),
            outputSampleRate);

        if (expectedRevision != 0 && expectedRevision != morphRevision_.load(std::memory_order_acquire))
        {
            error = "Morph request superseded.";
            morphedAudio.setSize(0, 0);
            return false;
        }

        MorphCacheEntry entry;
        entry.hash = cacheKey;
        entry.matchedTokens = std::move(matchedTokens);
        entry.audio = morphedAudio;
        entry.tokenChangePercent = tokenChangePercentOut;

        const juce::SpinLock::ScopedLockType lock(morphCacheMutex_);
        morphCache_.insert(morphCache_.begin(), std::move(entry));
        if (morphCache_.size() > maxMorphCacheEntries_)
            morphCache_.pop_back();
    }
    catch (const std::exception& ex)
    {
        error = ex.what();
        return false;
    }
    catch (...)
    {
        error = "Unknown render error.";
        return false;
    }

    return true;
}

void NeuralMorphingAudioProcessor::initialiseBackend()
{
    const auto codecId = selectedCodecId();
    const auto backendPolicy = selectedBackendPolicy();
    backendFallbackReason_.clear();
    activeBackendKind_ = ActiveBackendKind::Stub;

    const auto tryBridge = [&](const juce::String& reasonIfFail) -> bool
    {
#if NM_WITH_PYBRIDGE
        juce::String serverUrl = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_SERVER_URL", "http://localhost:8000");
        auto httpBackend = createHttpModelBackend(serverUrl);
        if (httpBackend == nullptr || !httpBackend->load(""))
        {
            backendFallbackReason_ = reasonIfFail + " | bridge unavailable";
            return false;
        }

        if (!httpBackend->setCodec(codecId.toStdString()))
        {
            backendFallbackReason_ = reasonIfFail + " | bridge codec switch failed";
            return false;
        }

        backend_ = std::move(httpBackend);
        activeBackendKind_ = ActiveBackendKind::BridgeHttp;
        return true;
#else
        juce::ignoreUnused(reasonIfFail);
        backendFallbackReason_ = "bridge backend is disabled at build time";
        return false;
#endif
    };

    const auto tryNativeDac = [&]() -> bool
    {
        if (codecId != "dac")
            return false;

#if NM_HAS_ONNX
        const auto modelOverride = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_MODEL_DIR", {});
        const juce::File modelRoot = modelOverride.isNotEmpty()
                                         ? juce::File(modelOverride)
                                         : juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                                               .getSiblingFile("dac-onnx");

        if (!modelRoot.isDirectory())
        {
            backendFallbackReason_ = "native DAC model path missing: " + modelRoot.getFullPathName();
            return false;
        }

        auto onnxBackend = createOnnxModelBackend();
        if (onnxBackend != nullptr && onnxBackend->load(modelRoot.getFullPathName().toStdString()))
        {
            backend_ = std::move(onnxBackend);
            activeBackendKind_ = ActiveBackendKind::NativeOnnx;
            return true;
        }

        backendFallbackReason_ = "native DAC load failed from " + modelRoot.getFullPathName();
        return false;
#else
        backendFallbackReason_ = "native DAC backend is disabled at build time";
        return false;
#endif
    };

    if (codecId == "spectrostream")
    {
        if (backendPolicy == BackendPolicy::NativeOnly)
        {
            backendFallbackReason_ = "spectrostream requires python bridge";
        }
        else
        {
            tryBridge("spectrostream selected");
        }
    }
    else
    {
        switch (backendPolicy)
        {
            case BackendPolicy::BridgeOnly:
                tryBridge("bridge_only policy");
                break;
            case BackendPolicy::NativeOnly:
                tryNativeDac();
                break;
            case BackendPolicy::NativePreferredBridgeFallback:
            default:
                if (!tryNativeDac())
                    tryBridge("native DAC unavailable");
                break;
        }
    }

    if (backend_ == nullptr)
    {
        if (backendFallbackReason_.isEmpty())
            backendFallbackReason_ = "no usable backend available";
    }

    if (backend_ != nullptr && !backend_->ready())
        backend_->load("");

    const int vectorDim = (backend_ != nullptr) ? juce::jmax(1, backend_->embeddingDimension()) : 2;
    paletteIndex_ = std::make_unique<PaletteIndex>(vectorDim);
    lastKnownProcessingModeParam_ = -1;
    applyProcessingModeChangeIfNeeded();
}

void NeuralMorphingAudioProcessor::switchBackend(int backendType)
{
    // Set the backend parameter
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("backend")))
        *param = juce::jlimit(0, 2, backendType);

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
    const int selectedCodec = juce::jlimit(0, 1, codecType);
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("bridgeCodec")))
        *param = selectedCodec;

    const auto defaults = selectedCodec == 1 ? kSpectroStreamDefaults : kDacDefaults;
    if (auto* p = dynamic_cast<juce::AudioParameterFloat*>(parameters.getParameter("temperature")))
        *p = defaults.temperature;
    if (auto* p = dynamic_cast<juce::AudioParameterFloat*>(parameters.getParameter("threshold")))
        *p = defaults.threshold;
    if (auto* p = dynamic_cast<juce::AudioParameterFloat*>(parameters.getParameter("continuity")))
        *p = defaults.continuity;
    if (auto* p = dynamic_cast<juce::AudioParameterFloat*>(parameters.getParameter("rvqFocus")))
        *p = defaults.rvqFocus;
    if (auto* p = dynamic_cast<juce::AudioParameterInt*>(parameters.getParameter("unit")))
        *p = defaults.unit;
    if (auto* p = dynamic_cast<juce::AudioParameterInt*>(parameters.getParameter("stride")))
        *p = defaults.stride;
    if (auto* p = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("swapMode")))
        *p = defaults.swapMode;

    shutdownWorkers();
    backend_.reset();
    paletteIndex_.reset();
    invalidateMorphCache();
    initialiseBackend();
    configureRealtimeTimings();
    createWorkers();
}

void NeuralMorphingAudioProcessor::applyDacDemoDefaults()
{
    if (selectedCodecId() != "dac")
        return;

    if (auto* p = dynamic_cast<juce::AudioParameterFloat*>(parameters.getParameter("threshold")))
    {
        if (p->get() < kDacDefaults.threshold)
            *p = kDacDefaults.threshold;
    }

    if (auto* p = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("swapMode")))
    {
        if (p->getIndex() < kDacDefaults.swapMode)
            *p = kDacDefaults.swapMode;
    }
}

void NeuralMorphingAudioProcessor::setProcessingMode(int modeType)
{
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(parameters.getParameter("processingMode")))
        *param = juce::jlimit(0, 1, modeType);
    applyProcessingModeChangeIfNeeded();
}

juce::String NeuralMorphingAudioProcessor::getBackendStatus() const
{
    if (backend_ == nullptr)
        return "No backend loaded";

    if (!backend_->ready())
        return "Backend not ready";

    juce::String wetState = "missing";
    switch (morphWetState_.load(std::memory_order_acquire))
    {
        case 1: wetState = "pending"; break;
        case 2: wetState = "ready"; break;
        case 3: wetState = "repeat"; break;
        case 4: wetState = "bypass"; break;
        default: break;
    }

    const int tokenChange = morphTokenChangePercent_.load(std::memory_order_acquire);
    juce::String tokenChangeText = tokenChange >= 0 ? juce::String(tokenChange) + "%" : "--";
    const auto fingerprint = morphAudioFingerprint_.load(std::memory_order_acquire);
    juce::String fingerprintText = fingerprint != 0 ? juce::String::toHexString(static_cast<int>(fingerprint)) : "--";
    const auto outputFingerprint = outputAudioFingerprint_.load(std::memory_order_acquire);
    juce::String outputFingerprintText = outputFingerprint != 0 ? juce::String::toHexString(static_cast<int>(outputFingerprint)) : "--";

    juce::String status = backendKindToString(activeBackendKind_)
                          + " | " + backendPolicyToString(selectedBackendPolicy())
                          + " | " + processingModeToString(selectedProcessingMode())
                          + " | " + selectedCodecId()
                          + " | wet=" + wetState + " " + juce::String(wetMixPercent_.load(std::memory_order_acquire)) + "%"
                          + " | tok=" + tokenChangeText
                          + " | sig=" + fingerprintText
                          + " | out=" + outputFingerprintText;

#if NM_WITH_PYBRIDGE
    if (auto* httpBackend = dynamic_cast<ModelBackendHttp*>(backend_.get()))
    {
        juce::String error = httpBackend->getLastError();
        if (error.isNotEmpty())
            status += " | err";
    }
#endif

    return status;
}

bool NeuralMorphingAudioProcessor::isBackendReady() const
{
    return backend_ != nullptr && backend_->ready();
}

bool NeuralMorphingAudioProcessor::renderStandaloneSourceToFile(const juce::File& outputFile, juce::String& error)
{
    error.clear();

    if (backend_ == nullptr || !backend_->ready())
    {
        error = "Backend is not ready.";
        return false;
    }

    if (!paletteReady())
    {
        error = "Palette is not ready.";
        return false;
    }

    if (paletteWorker_ != nullptr && paletteWorker_->isBusy())
    {
        error = "Palette is still rebuilding.";
        return false;
    }

    juce::AudioBuffer<float> source;
    double sourceSampleRate = currentSampleRate_ > 0.0 ? currentSampleRate_ : 44100.0;
    {
        const juce::ScopedLock lock(standaloneMutex_);
        if (!standaloneSourceLoaded_ || standaloneSourceBuffer_.getNumSamples() <= 0)
        {
            error = "No source sound loaded.";
            return false;
        }

        source.makeCopyOf(standaloneSourceBuffer_, true);
        if (standaloneSourceSampleRate_ > 0.0)
            sourceSampleRate = standaloneSourceSampleRate_;
    }

    const bool restartRealtimeWorker = realtimeWorkerThread_.joinable();
    stopRealtimeWorker();

    bool ok = false;
    do
    {
        const int requiredChannels = juce::jmax(1, backend_->requiredInputChannels());
        juce::AudioBuffer<float> modelInput(requiredChannels, source.getNumSamples());
        modelInput.clear();

        if (requiredChannels == 1)
        {
            const int sourceChannels = juce::jmax(1, source.getNumChannels());
            for (int ch = 0; ch < sourceChannels; ++ch)
                modelInput.addFrom(0, 0, source, ch, 0, source.getNumSamples(), 1.0f / static_cast<float>(sourceChannels));
        }
        else
        {
            for (int ch = 0; ch < requiredChannels; ++ch)
            {
                const int sourceCh = juce::jmin(ch, source.getNumChannels() - 1);
                modelInput.copyFrom(ch, 0, source, sourceCh, 0, source.getNumSamples());
            }
        }

        juce::AudioBuffer<float> morphedAudio;
        int tokenChange = -1;
        if (!renderMorphedAudioForInput(modelInput,
                                        sourceSampleRate,
                                        sourceSampleRate,
                                        morphedAudio,
                                        tokenChange,
                                        error))
            break;

        juce::AudioBuffer<float> rendered;
        rendered.makeCopyOf(source, true);
        juce::AudioBuffer<float> dry;
        dry.makeCopyOf(source, true);

        morphTokenChangePercent_.store(tokenChange, std::memory_order_release);
        morphAudioFingerprint_.store(audioFingerprint(morphedAudio), std::memory_order_release);
        mixOfflineMorphedAudio(rendered,
                              dry,
                              morphedAudio,
                              getParam("dryWet"),
                              getParam("envelopeFollow"),
                              sourceSampleRate);

        const float outputGain = juce::Decibels::decibelsToGain(getParam("outputGain"));
        rendered.applyGain(outputGain);
        applyOfflineSafetyLimiter(rendered);
        outputAudioFingerprint_.store(audioFingerprint(rendered), std::memory_order_release);

        if (!outputFile.getParentDirectory().createDirectory())
        {
            error = "Could not create output directory.";
            break;
        }

        if (outputFile.existsAsFile() && !outputFile.deleteFile())
        {
            error = "Could not overwrite output file.";
            break;
        }

        juce::WavAudioFormat wavFormat;
        auto stream = outputFile.createOutputStream();
        if (stream == nullptr)
        {
            error = "Could not open output file.";
            break;
        }

        std::unique_ptr<juce::AudioFormatWriter> writer(wavFormat.createWriterFor(
            stream.release(),
            sourceSampleRate,
            static_cast<unsigned int>(juce::jmax(1, rendered.getNumChannels())),
            24,
            {},
            0));

        if (writer == nullptr)
        {
            error = "Could not create WAV writer.";
            break;
        }

        if (!writer->writeFromAudioSampleBuffer(rendered, 0, rendered.getNumSamples()))
        {
            error = "Could not write rendered audio.";
            break;
        }

        ok = true;
    } while (false);

    if (restartRealtimeWorker)
        startRealtimeWorker();

    return ok;
}

void NeuralMorphingAudioProcessor::armHighQualityRender()
{
    setProcessingMode(0);
    invalidateMorphCache();
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralMorphingAudioProcessor();
}
