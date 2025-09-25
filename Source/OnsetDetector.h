#pragma once

#include <cstddef>

#include <juce_audio_basics/juce_audio_basics.h>

#include "LockFreeRing.h"

struct OnsetEvent
{
    int startSample = 0;
    int endSample = 0;
};

class OnsetDetector
{
public:
    OnsetDetector();

    void prepare(double sampleRate, int windowSize, int hopSize);
    void reset();

    void pushSamples(const juce::AudioBuffer<float>& monoBuffer);
    bool popEvent(OnsetEvent& event);

    void setThreshold(float newThreshold) { threshold_ = newThreshold; }

private:
    void processSample(float sampleValue);

    double sampleRate_ = 44100.0;
    int windowSize_ = 256;
    int hopSize_ = 128;
    int hopCounter_ = 0;

    float prevEnergy_ = 0.0f;
    float runningEnergy_ = 0.0f;
    float threshold_ = 0.35f;
    bool inEvent_ = false;
    int eventStart_ = 0;
    int processedSamples_ = 0;

    LockFreeRing<OnsetEvent> events_;
};
