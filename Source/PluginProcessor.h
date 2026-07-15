#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>

#include "LockFreeRing.h"
#include "ModelBackend.h"
#include "OnsetDetector.h"
#include "Workers.h"

class NeuralMorphingAudioProcessor : public juce::AudioProcessor,
                                     private juce::AudioProcessorValueTreeState::Listener
{
public:
    enum class ProcessingMode
    {
        QualityParity = 0,
        LiveRealtime
    };

    enum class BackendPolicy
    {
        NativePreferredBridgeFallback = 0,
        BridgeOnly,
        NativeOnly
    };

    enum class ActiveBackendKind
    {
        Stub = 0,
        NativeOnnx,
        BridgeHttp
    };

    NeuralMorphingAudioProcessor();
    ~NeuralMorphingAudioProcessor() override;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    //==============================================================================
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    //==============================================================================
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;
    void parameterChanged(const juce::String& parameterID, float newValue) override;

    float getParam(const juce::String& paramID) const;

    juce::AudioProcessorValueTreeState parameters;
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    PaletteWorker* getPaletteWorker() const { return paletteWorker_.get(); }
    MatchWorker* getMatchWorker() const { return matchWorker_.get(); }
    
    // Backend management
    void switchBackend(int backendType);
    void setBridgeCodec(int codecType);
    void setProcessingMode(int modeType);
    juce::String getBackendStatus() const;
    bool isBackendReady() const;
    float visualWetLevel() const { return visualWetLevel_.load(std::memory_order_acquire); }
    bool renderStandaloneSourceToFile(const juce::File& outputFile, juce::String& error);
    void flushDemoRealtimeCapture();
    void armHighQualityRender();

    void setStandaloneSource(juce::AudioBuffer<float> buffer, double sampleRate, const juce::String& name);
    void clearStandaloneSource();
    bool hasStandaloneSource() const;
    juce::String standaloneSourceName() const;

    void invalidateMorphCache(bool keepWet = false);

private:
    TokenBlock buildMatchedTokenBlock(const TokenBlock& targetBlock);
    void mixMorphedAudio(juce::AudioBuffer<float>& buffer, const juce::AudioBuffer<float>& dryBuffer, const juce::AudioBuffer<float>& morphed);
    bool paletteReady() const;
    bool isSilent(const juce::AudioBuffer<float>& buffer) const;

    bool isStandaloneWrapper() const;
    bool renderStandaloneSource(juce::AudioBuffer<float>& buffer);
    void resetMorphSmoothing();
    uint64_t hashTokenBlock(const TokenBlock& block) const;
    uint64_t hashMorphKey(const TokenBlock& block, double outputSampleRate) const;

    void applySafetyLimiter(juce::AudioBuffer<float>& buffer);
    void initialiseBackend();
    void shutdownWorkers();
    void createWorkers();
    void configureRealtimeTimings();
    int currentLatencySamplesForMode() const;
    void applyProcessingModeChangeIfNeeded();
    void clearRealtimeSessionState(bool clearHistoryBuffer);
    ProcessingMode selectedProcessingMode() const;
    BackendPolicy selectedBackendPolicy() const;
    juce::String selectedCodecId() const;
    void applyDacDemoDefaults();
    void updateWetAvailability(bool wetAvailable, int numSamples);
    juce::String backendKindToString(ActiveBackendKind kind) const;
    juce::String processingModeToString(ProcessingMode mode) const;
    juce::String backendPolicyToString(BackendPolicy policy) const;
    void pushRealtimeInputHistory(const juce::AudioBuffer<float>& inputBlock);
    const juce::AudioBuffer<float>& selectRealtimeEncodeInput(const juce::AudioBuffer<float>& currentBlock) const;
    void startRealtimeWorker();
    void stopRealtimeWorker();
    void queueRealtimeMorphTask(const juce::AudioBuffer<float>& encodeInput);
    void realtimeWorkerLoop();
    void processRealtimeMorphTokens(const TokenBlock& targetTokens,
                                    const juce::AudioBuffer<float>& sourceAudio,
                                    double outputSampleRate,
                                    uint64_t revision);
    bool renderMorphedAudioForInput(const juce::AudioBuffer<float>& encodeInput,
                                    double inputSampleRate,
                                    double outputSampleRate,
                                    juce::AudioBuffer<float>& morphedAudio,
                                    int& tokenChangePercentOut,
                                    juce::String& error,
                                    uint64_t expectedRevision = 0);
    bool renderMorphedTokenBlock(const TokenBlock& targetTokens,
                                 double outputSampleRate,
                                 juce::AudioBuffer<float>& morphedAudio,
                                 int& tokenChangePercentOut,
                                 juce::String& error,
                                 uint64_t expectedRevision = 0);

    std::unique_ptr<ModelBackend> backend_;
    std::unique_ptr<PaletteIndex> paletteIndex_;
    LockFreeRing<juce::AudioBuffer<float>> decodedFifo_;
    std::unique_ptr<PaletteWorker> paletteWorker_;
    std::unique_ptr<MatchWorker> matchWorker_;

    OnsetDetector onsetDetector_;
    juce::AudioBuffer<float> backendInputScratch_;
    juce::AudioBuffer<float> realtimeInputHistory_;
    juce::AudioBuffer<float> dryScratch_;

    double currentSampleRate_ = 44100.0;
    int samplesPerBlock_ = 0;
    bool isPrepared_ = false;

    std::vector<TokenBlock> targetSegments_;
    static constexpr std::size_t maxTargetSegments_ = 128;

    struct MorphCacheEntry
    {
        uint64_t hash = 0;
        TokenBlock matchedTokens;
        juce::AudioBuffer<float> audio;
        int tokenChangePercent = -1;
    };

    std::vector<MorphCacheEntry> morphCache_;
    static constexpr std::size_t maxMorphCacheEntries_ = 8;
    int morphUpdateIntervalMs_ = 40;
    int realtimeEncodeWindowMs_ = 0;
    int realtimeEncodeWindowSamples_ = 0;
    int qualityLookaheadMs_ = 1000;
    int qualitySuperframeSamples_ = 32768;
    int qualityHopSamples_ = 8192;
    int qualityCrossfadeSamples_ = 4096;
    int liveCandidateCount_ = 48;
    int liveBeamWidth_ = 6;
    int realtimeInputFilledSamples_ = 0;
    int currentLatencySamples_ = 2048;
    int lastKnownProcessingModeParam_ = -1;
    std::vector<float> sourceEnvelopeState_;
    std::vector<float> wetEnvelopeState_;
    std::vector<float> wetHighPassState1_;
    std::vector<float> wetHighPassState2_;
    mutable juce::SpinLock morphCacheMutex_;
    std::atomic<bool> resetSmoothingPending_{ false };
    std::atomic<uint64_t> morphRevision_{ 1 };
    std::atomic<bool> morphRefreshPending_{ false };
    std::atomic<bool> clearLatestTargetPending_{ false };
    std::atomic<int> lastMatchedIndex_{ -1 };
    int morphUpdateCountdownSamples_ = 0;
    juce::AudioBuffer<float> lastRealtimeMorphBlock_;
    juce::AudioBuffer<float> pendingRealtimeMorphBlock_;
    bool hasLastRealtimeMorphBlock_ = false;
    bool hasPendingRealtimeMorphBlock_ = false;
    int lastRealtimeMorphReadPosition_ = 0;
    bool lastRealtimeMorphWasUnderrun_ = false;
    int wetLoopStartPosition_ = 0;
    unsigned int wetLoopCycle_ = 0;
    std::vector<float> lastWetOutputSample_;
    std::vector<float> wetTransitionFromSample_;
    int wetTransitionRemainingSamples_ = 0;
    int wetTransitionTotalSamples_ = 0;
    bool wetTransitionPending_ = false;
    std::atomic<int> morphWetState_{ 0 };
    std::atomic<int> wetMixPercent_{ 0 };
    std::atomic<int> morphTokenChangePercent_{ -1 };
    std::atomic<uint32_t> morphAudioFingerprint_{ 0 };
    std::atomic<uint32_t> outputAudioFingerprint_{ 0 };
    std::atomic<float> visualWetLevel_{ 0.0f };
    float outputSafetyGain_ = 1.0f;
    float wetAvailabilityMix_ = 0.0f;
    juce::AudioBuffer<float> qualityTailBuffer_;
    bool qualityTailValid_ = false;
    std::thread realtimeWorkerThread_;
    std::atomic<bool> realtimeWorkerShouldExit_{ false };
    juce::WaitableEvent realtimeWorkerWake_;
    juce::CriticalSection realtimeTaskMutex_;
    juce::AudioBuffer<float> pendingRealtimeEncodeInput_;
    double pendingRealtimeSampleRate_ = 44100.0;
    bool hasPendingRealtimeTask_ = false;
    TokenBlock latestTargetTokens_;
    juce::AudioBuffer<float> latestTargetInput_;
    double latestTargetOutputSampleRate_ = 44100.0;

    mutable juce::CriticalSection standaloneMutex_;
    juce::AudioBuffer<float> standaloneSourceBuffer_;
    juce::String standaloneSourceName_;
    double standaloneSourceSampleRate_ = 0.0;
    int64_t standaloneSourcePosition_ = 0;
    bool standaloneSourceLoaded_ = false;
    bool deterministicDemoPlayback_ = false;

    juce::TimeSliceThread demoCaptureThread_{ "Realtime Capture" };
    std::unique_ptr<juce::AudioFormatWriter::ThreadedWriter> demoCaptureWriter_;
    std::atomic<int64_t> demoCaptureSamplesRemaining_{ 0 };

    ActiveBackendKind activeBackendKind_ = ActiveBackendKind::Stub;
    juce::String backendFallbackReason_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralMorphingAudioProcessor)
};
