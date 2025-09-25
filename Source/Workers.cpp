#include "Workers.h"

#include <cmath>

PaletteWorker::PaletteWorker(ModelBackend& backend, PaletteIndex& index)
    : juce::Thread("PaletteWorker"), backend_(backend), index_(index)
{
    formatManager_.registerBasicFormats();
    const juce::ScopedLock lock(stateMutex_);
    statusMessage_ = "Idle";
}

PaletteWorker::~PaletteWorker()
{
    signalThreadShouldExit();
    workReady_.signal();
    stopThread(2000);
}

void PaletteWorker::requestBuild(const std::vector<juce::File>& files, bool rebuildIndex)
{
    {
        const juce::ScopedLock lock(stateMutex_);
        pendingFiles_ = files;
    }

    rebuildRequested_.store(rebuildIndex, std::memory_order_release);
    progress_.store(0.0, std::memory_order_release);
    workReady_.signal();
}

juce::String PaletteWorker::status() const
{
    const juce::ScopedLock lock(stateMutex_);
    return statusMessage_;
}

void PaletteWorker::run()
{
    while (!threadShouldExit())
    {
        workReady_.wait(-1);
        if (threadShouldExit())
            break;

        processFiles();
    }
}

void PaletteWorker::processFiles()
{
    std::vector<juce::File> files;
    {
        const juce::ScopedLock lock(stateMutex_);
        files = pendingFiles_;
    }

    if (!backend_.ready())
    {
        const juce::ScopedLock lock(stateMutex_);
        statusMessage_ = "Model backend not ready";
        return;
    }

    if (files.empty())
    {
        index_.clear();
        const juce::ScopedLock lock(stateMutex_);
        statusMessage_ = "No palette files";
        return;
    }

    const bool rebuild = rebuildRequested_.exchange(false);
    juce::ignoreUnused(rebuild);

    busy_.store(true, std::memory_order_release);
    progress_.store(0.0, std::memory_order_release);
    index_.clear();

    for (size_t i = 0; i < files.size(); ++i)
    {
        if (threadShouldExit())
            break;

        auto file = files[i];
        std::unique_ptr<juce::AudioFormatReader> reader(formatManager_.createReaderFor(file));
        if (reader == nullptr)
        {
            const juce::ScopedLock lock(stateMutex_);
            statusMessage_ = "Failed to read " + file.getFileName();
            continue;
        }

        const int64 length = static_cast<int64>(reader->lengthInSamples);
        if (length <= 0)
            continue;

        juce::AudioBuffer<float> tempBuffer(static_cast<int>(reader->numChannels), static_cast<int>(length));
        reader->read(&tempBuffer, 0, static_cast<int>(length), 0, true, true);

        juce::AudioBuffer<float> monoBuffer(1, static_cast<int>(length));
        monoBuffer.clear();
        for (int ch = 0; ch < tempBuffer.getNumChannels(); ++ch)
            monoBuffer.addFrom(0, 0, tempBuffer, ch, 0, static_cast<int>(length), 1.0f / static_cast<float>(tempBuffer.getNumChannels()));

        auto tokens = backend_.encodePCM(monoBuffer);
        if (tokens.empty())
            continue;

        for (int frame = 0; frame < tokens.frames; ++frame)
        {
            const auto vectorRow = backend_.tokensToVectorRow(tokens, frame);
            if (static_cast<int>(vectorRow.size()) == index_.dimensions())
            {
                PaletteMeta meta;
                meta.sampleId = static_cast<int>(i);
                meta.frame = frame;
                index_.add(vectorRow, meta);
            }
        }

        progress_.store(static_cast<double>(i + 1) / static_cast<double>(files.size()), std::memory_order_release);
    }

    index_.build();
    busy_.store(false, std::memory_order_release);

    {
        const juce::ScopedLock lock(stateMutex_);
        statusMessage_ = juce::String("Indexed ") + juce::String(index_.size()) + " vectors";
    }
}

MatchWorker::MatchWorker(ModelBackend& backend, PaletteIndex& index, AudioBufferRing& outputRing)
    : juce::Thread("MatchWorker"), backend_(backend), index_(index), outputRing_(outputRing), taskQueue_(128)
{
}

MatchWorker::~MatchWorker()
{
    signalThreadShouldExit();
    workReady_.signal();
    stopThread(2000);
}

void MatchWorker::enqueue(const SegmentTask& task)
{
    if (!taskQueue_.push(task))
        return;

    workReady_.signal();
}

void MatchWorker::clearQueue()
{
    taskQueue_.clear();
}

void MatchWorker::run()
{
    while (!threadShouldExit())
    {
        workReady_.wait(-1);
        if (threadShouldExit())
            break;

        SegmentTask task;
        while (taskQueue_.pop(task))
            processTask(task);
    }
}

void MatchWorker::processTask(const SegmentTask& task)
{
    busy_.store(true, std::memory_order_release);

    const int duration = juce::jmax(backend_.sampleRate() / 20, task.event.endSample - task.event.startSample);
    juce::AudioBuffer<float> buffer(2, duration);
    buffer.clear();

    float baseFrequency = 220.0f;
    if (index_.size() > 0)
    {
        auto matches = index_.query({ 0.5f, 0.5f }, 1);
        if (!matches.empty())
            baseFrequency += static_cast<float>(matches.front().annIndex % 12) * 20.0f;
    }

    for (int sample = 0; sample < duration; ++sample)
    {
        const float t = static_cast<float>(sample) / static_cast<float>(backend_.sampleRate());
        const float envelope = std::exp(-4.0f * static_cast<float>(sample) / static_cast<float>(duration));
        const float value = std::sin(juce::MathConstants<float>::twoPi * baseFrequency * t) * envelope;
        buffer.setSample(0, sample, value);
        buffer.setSample(1, sample, value);
    }

    outputRing_.push(std::move(buffer));
    busy_.store(false, std::memory_order_release);
}

void Resampler::reset()
{
    interpolator_.reset();
    ratio_ = 1.0;
}

void Resampler::setRatio(double ratio)
{
    ratio_ = ratio;
}

void Resampler::process(const juce::AudioBuffer<float>& src, juce::AudioBuffer<float>& dst)
{
    const int numChannels = juce::jmin(src.getNumChannels(), dst.getNumChannels());
    const int numSamples = dst.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch)
        interpolator_.process(ratio_, src.getReadPointer(ch), dst.getWritePointer(ch), numSamples);
}
