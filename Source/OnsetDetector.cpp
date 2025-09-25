#include "OnsetDetector.h"

OnsetDetector::OnsetDetector()
    : events_(128)
{
}

void OnsetDetector::prepare(double sampleRate, int windowSize, int hopSize)
{
    sampleRate_ = sampleRate;
    windowSize_ = windowSize;
    hopSize_ = hopSize;
    reset();
}

void OnsetDetector::reset()
{
    hopCounter_ = 0;
    prevEnergy_ = 0.0f;
    runningEnergy_ = 0.0f;
    inEvent_ = false;
    eventStart_ = 0;
    processedSamples_ = 0;
    events_.clear();
}

void OnsetDetector::pushSamples(const juce::AudioBuffer<float>& monoBuffer)
{
    const float* data = monoBuffer.getReadPointer(0);
    for (int i = 0; i < monoBuffer.getNumSamples(); ++i)
        processSample(data[i]);
}

bool OnsetDetector::popEvent(OnsetEvent& event)
{
    return events_.pop(event);
}

void OnsetDetector::processSample(float sampleValue)
{
    const float absSample = std::abs(sampleValue);
    runningEnergy_ += absSample * absSample;
    ++hopCounter_;
    ++processedSamples_;

    if (hopCounter_ >= hopSize_)
    {
        const float energy = runningEnergy_ / static_cast<float>(windowSize_);
        const float delta = energy - prevEnergy_;
        prevEnergy_ = energy;
        hopCounter_ = 0;
        runningEnergy_ = 0.0f;

        if (!inEvent_ && delta > threshold_)
        {
            inEvent_ = true;
            eventStart_ = processedSamples_;
        }
        else if (inEvent_ && delta < threshold_ * 0.25f)
        {
            inEvent_ = false;
            const int eventEnd = processedSamples_;
            events_.push(OnsetEvent{ eventStart_, eventEnd });
        }
    }
}
