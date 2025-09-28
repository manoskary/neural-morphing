#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "JuceHeader.h"
#if NM_HAS_ONNX
#include "ModelBackendOnnx.h"
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

    paletteWorker_ = std::make_unique<PaletteWorker>(*backend_, *paletteIndex_);
    matchWorker_ = std::make_unique<MatchWorker>(*backend_, *paletteIndex_, decodedFifo_);

    monoScratch_.setSize(1, monoScratchReserve);
}

NeuralMorphingAudioProcessor::~NeuralMorphingAudioProcessor()
{
    if (paletteWorker_ != nullptr)
    {
        paletteWorker_->signalThreadShouldExit();
        paletteWorker_->stopThread(2000);
    }

    if (matchWorker_ != nullptr)
    {
        matchWorker_->signalThreadShouldExit();
        matchWorker_->stopThread(2000);
    }
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

    if (paletteWorker_ != nullptr && !paletteWorker_->isThreadRunning())
        paletteWorker_->startThread();

    if (matchWorker_ != nullptr && !matchWorker_->isThreadRunning())
        matchWorker_->startThread();

    setLatencySamples(2048);
}

void NeuralMorphingAudioProcessor::releaseResources()
{
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

        onsetDetector_.pushSamples(monoScratch_);

        OnsetEvent evt;
        while (onsetDetector_.popEvent(evt))
        {
            SegmentTask task;
            task.event = evt;
            task.sampleRate = static_cast<int>(currentSampleRate_);
            if (matchWorker_ != nullptr)
                matchWorker_->enqueue(task);
        }

        juce::AudioBuffer<float> decoded;
        while (decodedFifo_.pop(decoded))
        {
            if (decoded.getNumChannels() == 0 || decoded.getNumSamples() == 0)
                continue;

            const int copySamples = juce::jmin(decoded.getNumSamples(), buffer.getNumSamples());
            for (int ch = 0; ch < totalNumOutputChannels; ++ch)
            {
                const int srcCh = juce::jmin(ch, decoded.getNumChannels() - 1);
                buffer.addFrom(ch, 0, decoded, srcCh, 0, copySamples);
            }
        }

        const float dryWet = getParam("dryWet");
        buffer.applyGain(dryWet);
        for (int ch = 0; ch < totalNumOutputChannels; ++ch)
            buffer.addFrom(ch, 0, dryBuffer, ch, 0, numSamples, 1.0f - dryWet);
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

void NeuralMorphingAudioProcessor::initialiseBackend()
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

    if (backend_ == nullptr)
        backend_ = createStubModelBackend();

    if (backend_ != nullptr && !backend_->ready())
        backend_->load("");

    const int vectorDim = (backend_ != nullptr) ? juce::jmax(1, backend_->embeddingDimension()) : 2;
    paletteIndex_ = std::make_unique<PaletteIndex>(vectorDim);
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralMorphingAudioProcessor();
}
