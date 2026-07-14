#include "Workers.h"

#include <cmath>
#include <exception>

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

void PaletteWorker::requestBuild(const std::vector<juce::File>& files, bool rebuildIndex, int unit, int stride)
{
    juce::ignoreUnused(rebuildIndex);
    {
        const juce::ScopedLock lock(stateMutex_);
        pendingFiles_ = files;
        buildUnit_ = juce::jmax(1, unit);
        buildStride_ = juce::jlimit(1, buildUnit_, stride);
    }

    buildRequested_.store(true, std::memory_order_release);
    progress_.store(0.0, std::memory_order_release);
    workReady_.signal();
}

void PaletteWorker::requestRegrain(int unit, int stride)
{
    const int safeUnit = juce::jmax(1, unit);
    requestedUnit_.store(safeUnit, std::memory_order_release);
    requestedStride_.store(juce::jlimit(1, safeUnit, stride), std::memory_order_release);
    regrainRequested_.store(true, std::memory_order_release);
    workReady_.signal();
}

juce::String PaletteWorker::status() const
{
    const juce::ScopedLock lock(stateMutex_);
    return statusMessage_;
}

void PaletteWorker::shutdown()
{
    workReady_.signal();
}

void PaletteWorker::run()
{
    while (!threadShouldExit())
    {
        workReady_.wait(-1);
        if (threadShouldExit())
            break;

        try
        {
            if (buildRequested_.exchange(false, std::memory_order_acq_rel))
                processFiles();
            if (regrainRequested_.exchange(false, std::memory_order_acq_rel))
                processRegrain();
        }
        catch (const std::exception& error)
        {
            busy_.store(false, std::memory_order_release);
            const juce::ScopedLock lock(stateMutex_);
            statusMessage_ = "Palette error: " + juce::String(error.what());
        }
        catch (...)
        {
            busy_.store(false, std::memory_order_release);
            const juce::ScopedLock lock(stateMutex_);
            statusMessage_ = "Palette error: unknown backend failure";
        }
    }
}

void PaletteWorker::processFiles()
{
    std::vector<juce::File> files;
    int unit = 1;
    int stride = 1;
    {
        const juce::ScopedLock lock(stateMutex_);
        files = pendingFiles_;
        unit = buildUnit_;
        stride = buildStride_;
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

    busy_.store(true, std::memory_order_release);
    progress_.store(0.0, std::memory_order_release);

    constexpr int variantsPerFile = 3;
    index_.prepareForSamples(static_cast<int>(files.size()) * variantsPerFile);

    {
        const juce::ScopedLock lock(stateMutex_);
            statusMessage_ = "Indexing " + juce::String(files.size()) + " palette sounds";
    }

    int sampleId = 0;
    bool trimmedLongFile = false;
    auto indexPaletteVariant = [&](juce::AudioBuffer<float>& modelInput, double sourceSampleRate) -> bool
    {
        auto tokens = backend_.encodePCM(modelInput, sourceSampleRate);
        if (tokens.empty())
            return false;

        const int currentSampleId = sampleId++;
        std::vector<std::vector<float>> frameVectors;
        const bool hasFrameVectors = backend_.tokensToVectorRows(tokens, 0, tokens.frames, frameVectors)
                                     && static_cast<int>(frameVectors.size()) == tokens.frames;
        if (!hasFrameVectors)
            return false;

        index_.setSampleData(currentSampleId, std::move(tokens), std::move(frameVectors));

        return true;
    };

    for (size_t i = 0; i < files.size(); ++i)
    {
        if (threadShouldExit())
            break;

        auto file = files[i];
        {
            const juce::ScopedLock lock(stateMutex_);
            statusMessage_ = "Indexing palette sound " + file.getFileName() + " (" + juce::String(i + 1) + "/" + juce::String(files.size()) + ")";
        }
        std::unique_ptr<juce::AudioFormatReader> reader(formatManager_.createReaderFor(file));
        if (reader == nullptr)
        {
            const juce::ScopedLock lock(stateMutex_);
            statusMessage_ = "Failed to read " + file.getFileName();
            continue;
        }

        const juce::int64 availableLength = static_cast<juce::int64>(reader->lengthInSamples);
        if (availableLength <= 0 || reader->sampleRate <= 0.0)
            continue;

        constexpr double maxPaletteSeconds = 10.0;
        const auto maxLength = static_cast<juce::int64>(std::ceil(reader->sampleRate * maxPaletteSeconds));
        const int length = static_cast<int>(juce::jmin(availableLength, maxLength));
        trimmedLongFile = trimmedLongFile || availableLength > maxLength;

        juce::AudioBuffer<float> tempBuffer(static_cast<int>(reader->numChannels), length);
        if (!reader->read(&tempBuffer, 0, length, 0, true, true))
            continue;

        const int requiredChannels = juce::jmax(1, backend_.requiredInputChannels());
        juce::AudioBuffer<float> modelInput(requiredChannels, length);
        modelInput.clear();

        if (requiredChannels == 1)
        {
            for (int ch = 0; ch < tempBuffer.getNumChannels(); ++ch)
                modelInput.addFrom(0, 0, tempBuffer, ch, 0, length, 1.0f / static_cast<float>(tempBuffer.getNumChannels()));
        }
        else
        {
            const int srcChannels = juce::jmax(1, tempBuffer.getNumChannels());
            for (int ch = 0; ch < requiredChannels; ++ch)
            {
                const int srcCh = juce::jmin(ch, srcChannels - 1);
                modelInput.copyFrom(ch, 0, tempBuffer, srcCh, 0, length);
            }
        }

        indexPaletteVariant(modelInput, reader->sampleRate);

        for (float gain : { 0.7f, 0.3f })
        {
            juce::AudioBuffer<float> augmented;
            augmented.makeCopyOf(modelInput, true);
            augmented.applyGain(gain);
            indexPaletteVariant(augmented, reader->sampleRate);
        }

        progress_.store(static_cast<double>(i + 1) / static_cast<double>(files.size()), std::memory_order_release);
    }

    const bool published = index_.publishBuild(unit, stride);
    busy_.store(false, std::memory_order_release);

    {
        const juce::ScopedLock lock(stateMutex_);
        statusMessage_ = published
                             ? juce::String("Indexed ") + juce::String(index_.size()) + " vectors"
                                   + (trimmedLongFile ? " (long files use first 10 s)" : "")
                             : "Palette produced no valid grains";
    }
}

void PaletteWorker::processRegrain()
{
    busy_.store(true, std::memory_order_release);
    const int unit = requestedUnit_.load(std::memory_order_acquire);
    const int stride = requestedStride_.load(std::memory_order_acquire);
    const bool rebuilt = index_.rebuildGrains(unit, stride);
    busy_.store(false, std::memory_order_release);

    const juce::ScopedLock lock(stateMutex_);
    statusMessage_ = rebuilt
                         ? juce::String("Regrained ") + juce::String(index_.size()) + " vectors"
                         : "Palette is not ready for regraining";
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

void MatchWorker::shutdown()
{
    clearQueue();
    workReady_.signal();
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
        const auto snapshot = index_.snapshot();
        std::vector<float> query(static_cast<size_t>(index_.dimensions()), 0.5f);
        RvqSearchConfig search;
        search.coarseLength = index_.dimensions();
        auto matches = index_.query(snapshot, query, 1, search);
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
