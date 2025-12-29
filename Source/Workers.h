#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>
#include <juce_dsp/juce_dsp.h>

#include "LockFreeRing.h"
#include "ModelBackend.h"
#include "OnsetDetector.h"
#include "PaletteIndex.h"

struct SegmentTask
{
    OnsetEvent event;
    int sampleRate = 44100;
};

class PaletteWorker : public juce::Thread
{
public:
    PaletteWorker(ModelBackend& backend, PaletteIndex& index);
    ~PaletteWorker() override;

    void requestBuild(const std::vector<juce::File>& files, bool rebuildIndex, int unit, int stride);
    bool isBusy() const noexcept { return busy_.load(); }
    double progress() const noexcept { return progress_.load(); }
    juce::String status() const;
    void shutdown();

protected:
    void run() override;

private:
    void processFiles();

    ModelBackend& backend_;
    PaletteIndex& index_;
    juce::AudioFormatManager formatManager_;
    std::vector<juce::File> pendingFiles_;
    int buildUnit_ = 1;
    int buildStride_ = 1;

    std::atomic<bool> rebuildRequested_{ false };
    std::atomic<bool> busy_{ false };
    std::atomic<double> progress_{ 0.0 };
    juce::WaitableEvent workReady_;
    juce::String statusMessage_;
    juce::CriticalSection stateMutex_;
};

class MatchWorker : public juce::Thread
{
public:
    using AudioBufferRing = LockFreeRing<juce::AudioBuffer<float>>;

    MatchWorker(ModelBackend& backend, PaletteIndex& index, AudioBufferRing& outputRing);
    ~MatchWorker() override;

    void enqueue(const SegmentTask& task);
    bool isBusy() const noexcept { return busy_.load(); }
    void clearQueue();
    void shutdown();

protected:
    void run() override;

private:
    void processTask(const SegmentTask& task);

    ModelBackend& backend_;
    PaletteIndex& index_;
    AudioBufferRing& outputRing_;
    LockFreeRing<SegmentTask> taskQueue_;
    std::atomic<bool> busy_{ false };
    juce::WaitableEvent workReady_;
};

class Resampler
{
public:
    void reset();
    void setRatio(double ratio);
    void process(const juce::AudioBuffer<float>& src, juce::AudioBuffer<float>& dst);

private:
    double ratio_ = 1.0;
    juce::LagrangeInterpolator interpolator_;
};
