#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>

#include "LockFreeRing.h"
#include "ModelBackend.h"
#include "OnsetDetector.h"
#include "Workers.h"

class NeuralMorphingAudioProcessor : public juce::AudioProcessor
{
public:
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

    float getParam(const juce::String& paramID) const;

    juce::AudioProcessorValueTreeState parameters;
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    PaletteWorker* getPaletteWorker() const { return paletteWorker_.get(); }
    MatchWorker* getMatchWorker() const { return matchWorker_.get(); }
    
    // Backend management
    void switchBackend(int backendType);
    juce::String getBackendStatus() const;
    bool isBackendReady() const;

    void setStandaloneSource(juce::AudioBuffer<float> buffer, double sampleRate, const juce::String& name);
    void clearStandaloneSource();
    bool hasStandaloneSource() const;
    juce::String standaloneSourceName() const;

    void invalidateMorphCache();

private:
    TokenBlock buildMatchedTokenBlock(const TokenBlock& targetBlock);
    void mixMorphedAudio(juce::AudioBuffer<float>& buffer, const juce::AudioBuffer<float>& dryBuffer, const juce::AudioBuffer<float>& morphed);
    bool paletteReady() const;
    bool isSilent(const juce::AudioBuffer<float>& buffer) const;

    bool isStandaloneWrapper() const;
    bool renderStandaloneSource(juce::AudioBuffer<float>& buffer);
    void resetMorphSmoothing();
    uint64_t hashTokenBlock(const TokenBlock& block) const;

    void refreshBackendSampleRate(double sampleRate);
    void mixWetBuffer(juce::AudioBuffer<float>& buffer, juce::AudioBuffer<float>& dryBuffer);
    void initialiseBackend();
    void shutdownWorkers();
    void createWorkers();

    std::unique_ptr<ModelBackend> backend_;
    std::unique_ptr<PaletteIndex> paletteIndex_;
    LockFreeRing<juce::AudioBuffer<float>> decodedFifo_;
    std::unique_ptr<PaletteWorker> paletteWorker_;
    std::unique_ptr<MatchWorker> matchWorker_;

    OnsetDetector onsetDetector_;
    juce::AudioBuffer<float> monoScratch_;
    juce::AudioBuffer<float> morphScratch_;

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
    };

    std::vector<MorphCacheEntry> morphCache_;
    static constexpr std::size_t maxMorphCacheEntries_ = 8;
    std::vector<float> morphSmoothingState_;
    mutable juce::SpinLock morphCacheMutex_;
    std::atomic<bool> resetSmoothingPending_{ false };

    mutable juce::CriticalSection standaloneMutex_;
    juce::AudioBuffer<float> standaloneSourceBuffer_;
    juce::String standaloneSourceName_;
    double standaloneSourceSampleRate_ = 0.0;
    int64_t standaloneSourcePosition_ = 0;
    bool standaloneSourceLoaded_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralMorphingAudioProcessor)
};
