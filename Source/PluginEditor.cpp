#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "JuceHeader.h"

NeuralMorphingAudioProcessorEditor::NeuralMorphingAudioProcessorEditor(NeuralMorphingAudioProcessor& p)
    : juce::AudioProcessorEditor(&p), processor_(p)
{
    setSize(640, 420); // Increased height to accommodate backend selector

    addAndMakeVisible(loadButton_);
    addAndMakeVisible(clearButton_);
    addAndMakeVisible(rebuildButton_);
    addAndMakeVisible(statusLabel_);
    addAndMakeVisible(progressLabel_);

    setupSlider(temperatureSlider_, "Temperature");
    setupSlider(thresholdSlider_, "Threshold");
    setupSlider(unitSlider_, "Unit");
    setupSlider(strideSlider_, "Stride");
    setupSlider(similaritySlider_, "Similarity");
    setupSlider(envelopeSlider_, "Envelope");
    setupSlider(dryWetSlider_, "Dry/Wet");
    setupSlider(outputSlider_, "Output");

    temperatureAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "temperature", temperatureSlider_);
    thresholdAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "threshold", thresholdSlider_);
    unitAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "unit", unitSlider_);
    strideAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "stride", strideSlider_);
    similarityAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "similarity", similaritySlider_);
    envelopeAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "envelopeFollow", envelopeSlider_);
    dryWetAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "dryWet", dryWetSlider_);
    outputAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "outputGain", outputSlider_);

    // Setup backend selector
    addAndMakeVisible(backendSelector_);
    addAndMakeVisible(backendLabel_);
    addAndMakeVisible(statusDisplayLabel_);
    
    backendSelector_.addItem("Native", 1);
#if NM_WITH_PYBRIDGE
    backendSelector_.addItem("Python Bridge", 2);
#endif
    
    backendAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(processor_.parameters, "backend", backendSelector_);
    
    backendLabel_.attachToComponent(&backendSelector_, true);
    statusDisplayLabel_.setJustificationType(juce::Justification::centredLeft);
    statusDisplayLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgrey);

    statusLabel_.setJustificationType(juce::Justification::centredLeft);
    progressLabel_.setJustificationType(juce::Justification::centredLeft);

    loadButton_.addListener(this);
    clearButton_.addListener(this);
    rebuildButton_.addListener(this);
    backendSelector_.addListener(this);

    startTimerHz(10);
}

NeuralMorphingAudioProcessorEditor::~NeuralMorphingAudioProcessorEditor()
{
    stopTimer();
    loadButton_.removeListener(this);
    clearButton_.removeListener(this);
    rebuildButton_.removeListener(this);
    backendSelector_.removeListener(this);
}

void NeuralMorphingAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(16.0f, juce::Font::bold));
    g.drawText("Neural Morphing", 20, 10, getWidth() - 40, 24, juce::Justification::centredLeft);
}

void NeuralMorphingAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced(12);

    auto headerArea = area.removeFromTop(32);
    loadButton_.setBounds(headerArea.removeFromLeft(160).reduced(2));
    clearButton_.setBounds(headerArea.removeFromLeft(120).reduced(2));
    rebuildButton_.setBounds(headerArea.removeFromLeft(120).reduced(2));

    auto statusArea = area.removeFromTop(24);
    statusLabel_.setBounds(statusArea.removeFromLeft(getWidth() / 2));
    progressLabel_.setBounds(statusArea);

    // Backend selector area
    auto backendArea = area.removeFromTop(32);
    backendSelector_.setBounds(backendArea.removeFromLeft(200).reduced(2));
    
    // Backend status area
    auto backendStatusArea = area.removeFromTop(24);
    statusDisplayLabel_.setBounds(backendStatusArea.reduced(2));

    auto sliderArea = area.reduced(0, 10);
    const int numColumns = 4;
    const int numRows = 2;
    const int sliderWidth = sliderArea.getWidth() / numColumns;
    const int sliderHeight = sliderArea.getHeight() / numRows;

    juce::Slider* sliders[] = { &temperatureSlider_, &thresholdSlider_, &unitSlider_, &strideSlider_,
                                &similaritySlider_, &envelopeSlider_, &dryWetSlider_, &outputSlider_ };

    for (int row = 0; row < numRows; ++row)
        for (int col = 0; col < numColumns; ++col)
        {
            const int index = row * numColumns + col;
            auto bounds = sliderArea.withTrimmedTop(row * sliderHeight).withTrimmedLeft(col * sliderWidth).removeFromTop(sliderHeight).removeFromLeft(sliderWidth).reduced(6);
            sliders[index]->setBounds(bounds);
        }
}

void NeuralMorphingAudioProcessorEditor::buttonClicked(juce::Button* button)
{
    if (button == &loadButton_)
    {
        juce::String initialPath = lastDirectory_.exists() ? lastDirectory_.getFullPathName() : juce::File::getSpecialLocation(juce::File::userHomeDirectory).getFullPathName();
        
        // Use the asynchronous version for better compatibility
        auto chooser = std::make_unique<juce::FileChooser>("Select palette audio", juce::File(initialPath), "*.wav;*.flac;*.mp3;*.aiff;*.ogg");
        chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                           [this, chooserPtr = chooser.get()](const juce::FileChooser& fc)
                           {
                               auto result = fc.getResult();
                               if (result != juce::File{})
                               {
                                   lastFiles_.clear();
                                   lastFiles_.push_back(result);

                                   if (!lastFiles_.empty())
                                   {
                                       lastDirectory_ = lastFiles_.front().getParentDirectory();
                                       if (auto* worker = processor_.getPaletteWorker())
                                           worker->requestBuild(lastFiles_, true);
                                   }
                               }
                           });
        chooser.release(); // FileChooser will manage its own lifetime
    }
    else if (button == &clearButton_)
    {
        lastFiles_.clear();
        if (auto* worker = processor_.getPaletteWorker())
            worker->requestBuild({}, true);
    }
    else if (button == &rebuildButton_)
    {
        if (auto* worker = processor_.getPaletteWorker())
            worker->requestBuild(lastFiles_, true);
    }
}

void NeuralMorphingAudioProcessorEditor::comboBoxChanged(juce::ComboBox* comboBox)
{
    if (comboBox == &backendSelector_)
    {
        int backendIndex = backendSelector_.getSelectedItemIndex();
        processor_.switchBackend(backendIndex);
    }
}

void NeuralMorphingAudioProcessorEditor::timerCallback()
{
    juce::String statusText = "Idle";
    juce::String progressText;

    if (auto* worker = processor_.getPaletteWorker())
    {
        statusText = worker->status();
        if (worker->isBusy())
            progressText = juce::String(worker->progress() * 100.0, 1) + "%";
        else
            progressText = "Ready";
    }

    if (auto* match = processor_.getMatchWorker())
    {
        if (match->isBusy())
            progressText = "Matching";
    }

    statusLabel_.setText(statusText, juce::dontSendNotification);
    progressLabel_.setText(progressText, juce::dontSendNotification);
    
    // Update backend status
    juce::String backendStatus = processor_.getBackendStatus();
    statusDisplayLabel_.setText(backendStatus, juce::dontSendNotification);
}

void NeuralMorphingAudioProcessorEditor::setupSlider(juce::Slider& slider, const juce::String& name)
{
    slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 20);
    slider.setName(name);
    slider.setTooltip(name);
    addAndMakeVisible(slider);
}
