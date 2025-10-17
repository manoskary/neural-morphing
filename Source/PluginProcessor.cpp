#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "JuceHeader.h"
#if NM_HAS_ONNX
#include "ModelBackendOnnx.h"
#endif
#if NM_WITH_PYBRIDGE
#include "ModelBackendHttp.h"
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
    createWorkers();

    monoScratch_.setSize(1, monoScratchReserve);
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

    onsetDetector_.prepare(sampleRate, 512, 256);
    onsetDetector_.reset();

    isPrepared_ = true;

    if (paletteWorker_ != nullptr && !paletteWorker_->isThreadRunning())
        paletteWorker_->startThread();

    if (matchWorker_ != nullptr && !matchWorker_->isThreadRunning())
        matchWorker_->startThread();

    setLatencySamples(2048);
}

void NeuralMorphingAudioProcessor::releaseResources()
{
    isPrepared_ = false;
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

    // Backend selection (0=Native/ONNX, 1=Python Bridge)
    juce::StringArray backendChoices;
    backendChoices.add("Native");
#if NM_WITH_PYBRIDGE
    backendChoices.add("Python Bridge");
#endif
    params.push_back(std::make_unique<juce::AudioParameterChoice>("backend", "Backend", backendChoices, 0));

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
    matchWorker_ = std::make_unique<MatchWorker>(*backend_, *paletteIndex_, decodedFifo_);

    if (isPrepared_)
    {
        if (!paletteWorker_->isThreadRunning())
            paletteWorker_->startThread();

        if (!matchWorker_->isThreadRunning())
            matchWorker_->startThread();
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

    // Fallback to stub backend if nothing else worked
    if (backend_ == nullptr)
        backend_ = createStubModelBackend();

    if (backend_ != nullptr && !backend_->ready())
        backend_->load("");

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

    // Reinitialize with new backend
    initialiseBackend();
    createWorkers();
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
        return "HTTP Backend connected to " + httpBackend->getServerUrl();
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
