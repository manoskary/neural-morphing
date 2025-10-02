#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "JuceHeader.h"
#include "BinaryData.h"

namespace
{
const auto neonGreen = juce::Colour::fromRGB(0x3C, 0xFF, 0x9B);
const auto deepGrey = juce::Colour::fromRGB(0x16, 0x1E, 0x1F);
const auto darkSlate = juce::Colour::fromRGB(0x0B, 0x12, 0x14);
const auto accentGrey = juce::Colour::fromRGB(0x28, 0x32, 0x34);
const auto textGrey = juce::Colour::fromRGB(0xB0, 0xC8, 0xC3);
constexpr float knobStroke = 4.0f;
}

NeuralMorphingLookAndFeel::NeuralMorphingLookAndFeel()
{
    knobSprite_ = juce::ImageCache::getFromMemory(BinaryData::knob1_png, BinaryData::knob1_pngSize);

    setColour(juce::Slider::thumbColourId, neonGreen);
    setColour(juce::Slider::rotarySliderFillColourId, neonGreen);
    setColour(juce::Slider::rotarySliderOutlineColourId, accentGrey);
    setColour(juce::Slider::textBoxTextColourId, neonGreen.brighter(0.2f));
    setColour(juce::Slider::textBoxBackgroundColourId, darkSlate);
    setColour(juce::Slider::textBoxHighlightColourId, neonGreen.withAlpha(0.4f));

    setColour(juce::TextButton::buttonColourId, accentGrey);
    setColour(juce::TextButton::buttonOnColourId, neonGreen.darker(0.4f));
    setColour(juce::TextButton::textColourOffId, textGrey);
    setColour(juce::TextButton::textColourOnId, neonGreen);

    setColour(juce::ComboBox::backgroundColourId, accentGrey);
    setColour(juce::ComboBox::outlineColourId, juce::Colours::transparentBlack);
    setColour(juce::ComboBox::arrowColourId, neonGreen);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, neonGreen.withAlpha(0.22f));
    setColour(juce::PopupMenu::highlightedTextColourId, juce::Colours::black);

    setColour(juce::Label::textColourId, textGrey);
}

void NeuralMorphingLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                                 float sliderPosProportional, float rotaryStartAngle,
                                                 float rotaryEndAngle, juce::Slider& slider)
{
    auto bounds = juce::Rectangle<float>(static_cast<float>(x), static_cast<float>(y),
                                         static_cast<float>(width), static_cast<float>(height)).reduced(4.0f);

    const auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) * 0.5f;
    const auto centre = bounds.getCentre();

    // Fluorescent halo behind the knob to blend with the rest of the theme.
    juce::Colour haloColour = neonGreen.withAlpha(slider.isEnabled() ? 0.18f : 0.05f);
    g.setColour(haloColour);
    g.fillEllipse(centre.x - radius * 0.95f, centre.y - radius * 0.95f,
                  radius * 1.9f, radius * 1.9f);

    if (knobSprite_.isValid())
    {
        const int frameSize = knobSprite_.getWidth();
        const int frameCount = frameSize > 0 ? knobSprite_.getHeight() / frameSize : 0;

        if (frameSize > 0 && frameCount > 0)
        {
            const float sliderNorm = juce::jlimit(0.0f, 1.0f, sliderPosProportional);
            const int frameIndex = juce::jlimit(0, frameCount - 1,
                                                static_cast<int>(std::round(sliderNorm * (frameCount - 1))));

            const float renderSize = radius * 2.0f;
            const float drawX = centre.x - renderSize * 0.5f;
            const float drawY = centre.y - renderSize * 0.5f;

            // Soft shadow for depth
            g.setColour(juce::Colours::black.withAlpha(0.35f));
            g.fillEllipse(drawX + 2.0f, drawY + renderSize * 0.65f,
                          renderSize - 4.0f, renderSize * 0.4f);

            g.drawImage(knobSprite_,
                        static_cast<int>(drawX),
                        static_cast<int>(drawY),
                        static_cast<int>(renderSize),
                        static_cast<int>(renderSize),
                        0,
                        frameIndex * frameSize,
                        frameSize,
                        frameSize);
            return;
        }
    }

    // Fallback vector rendering if the sprite is unavailable.
    const auto angle = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

    juce::Colour outerRing = accentGrey;
    juce::Colour innerFill = deepGrey;
    juce::Colour valueColour = neonGreen;

    g.setGradientFill(juce::ColourGradient(innerFill.brighter(0.05f), centre.x, centre.y - radius * 0.8f,
                                           innerFill.darker(0.2f), centre.x, centre.y + radius * 0.8f, false));
    g.fillEllipse(bounds);

    g.setColour(outerRing);
    g.drawEllipse(bounds, knobStroke * 0.5f);

    juce::Path valueArc;
    valueArc.addCentredArc(centre.x, centre.y, radius, radius, 0.0f, rotaryStartAngle, angle, true);
    g.setColour(valueColour);
    g.strokePath(valueArc, juce::PathStrokeType(knobStroke, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    const auto pointerLength = radius * 0.72f;
    const auto pointerPos = centre.getPointOnCircumference(pointerLength, angle);
    g.setColour(valueColour.brighter(0.2f));
    g.fillEllipse(pointerPos.x - 6.0f, pointerPos.y - 6.0f, 12.0f, 12.0f);
}

void NeuralMorphingLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                                     const juce::Colour& backgroundColour,
                                                     bool isMouseOverButton, bool isButtonDown)
{
    auto bounds = button.getLocalBounds().toFloat();
    const float cornerSize = 8.0f;

    auto base = backgroundColour;
    if (isButtonDown)
        base = neonGreen.darker(0.4f);
    else if (isMouseOverButton)
        base = base.brighter(0.2f);

    g.setGradientFill({ base.brighter(0.1f), bounds.getTopLeft(),
                        base.darker(0.25f), bounds.getBottomRight(), false });
    g.fillRoundedRectangle(bounds, cornerSize);

    g.setColour(base.darker(0.6f));
    g.drawRoundedRectangle(bounds, cornerSize, 1.5f);
}

void NeuralMorphingLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool,
                                             int, int, int, int, juce::ComboBox& box)
{
    const auto bounds = juce::Rectangle<int>(0, 0, width, height);
    g.setColour(accentGrey);
    g.fillRoundedRectangle(bounds.toFloat(), 6.0f);

    g.setColour(accentGrey.darker(0.7f));
    g.drawRoundedRectangle(bounds.toFloat(), 6.0f, 1.5f);

    g.setColour(neonGreen);
    const juce::Point<float> arrowPoint(float(width) - 18.0f, float(height) * 0.5f);
    juce::Path arrow;
    arrow.addTriangle(arrowPoint.x - 6.0f, arrowPoint.y - 3.0f,
                      arrowPoint.x + 6.0f, arrowPoint.y - 3.0f,
                      arrowPoint.x, arrowPoint.y + 5.0f);
    g.fillPath(arrow);
}

juce::Font NeuralMorphingLookAndFeel::getComboBoxFont(juce::ComboBox&)
{
    return juce::Font(14.0f, juce::Font::bold);
}

NeuralMorphingAudioProcessorEditor::NeuralMorphingAudioProcessorEditor(NeuralMorphingAudioProcessor& p)
    : juce::AudioProcessorEditor(&p), processor_(p)
{
    setLookAndFeel(&lookAndFeel_);
    setSize(640, 440);

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
    backendLabel_.setColour(juce::Label::textColourId, textGrey);
    statusDisplayLabel_.setJustificationType(juce::Justification::centredLeft);
    statusDisplayLabel_.setColour(juce::Label::textColourId, neonGreen.withAlpha(0.8f));

    statusLabel_.setJustificationType(juce::Justification::centredLeft);
    progressLabel_.setJustificationType(juce::Justification::centredRight);
    statusLabel_.setColour(juce::Label::textColourId, textGrey);
    progressLabel_.setColour(juce::Label::textColourId, neonGreen.withAlpha(0.9f));

    loadButton_.setColour(juce::TextButton::buttonColourId, accentGrey);
    clearButton_.setColour(juce::TextButton::buttonColourId, accentGrey);
    rebuildButton_.setColour(juce::TextButton::buttonColourId, accentGrey);

    loadButton_.addListener(this);
    clearButton_.addListener(this);
    rebuildButton_.addListener(this);
    backendSelector_.addListener(this);

    startTimerHz(10);
}

NeuralMorphingAudioProcessorEditor::~NeuralMorphingAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
    loadButton_.removeListener(this);
    clearButton_.removeListener(this);
    rebuildButton_.removeListener(this);
    backendSelector_.removeListener(this);
}

void NeuralMorphingAudioProcessorEditor::paint(juce::Graphics& g)
{
    juce::ColourGradient gradient(darkSlate, 0.0f, 0.0f,
                                  deepGrey.darker(0.4f), 0.0f, static_cast<float>(getHeight()), false);
    gradient.addColour(0.5f, deepGrey);
    g.setGradientFill(gradient);
    g.fillAll();

    juce::Rectangle<int> headerBounds = { 20, 10, getWidth() - 40, 28 };
    g.setColour(neonGreen);
    g.setFont(juce::Font(22.0f, juce::Font::bold));
    g.drawText("Neural Morphing", headerBounds, juce::Justification::centredLeft);
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
            if (index >= static_cast<int>(std::size(sliders)))
                continue;

            juce::Rectangle<int> cell(sliderArea.getX() + col * sliderWidth,
                                      sliderArea.getY() + row * sliderHeight,
                                      sliderWidth,
                                      sliderHeight);

            auto labelBounds = cell.removeFromTop(24).reduced(8, 2);
            auto sliderBounds = cell.reduced(12, 6);

            sliders[index]->setBounds(sliderBounds);
            if (index < static_cast<int>(sliderLabels_.size()))
                sliderLabels_[index]->setBounds(labelBounds);
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
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::trackColourId, accentGrey);
    addAndMakeVisible(slider);

    auto label = std::make_unique<juce::Label>();
    label->setText(name, juce::dontSendNotification);
    label->setJustificationType(juce::Justification::centred);
    label->setColour(juce::Label::textColourId, textGrey);
    label->setFont(juce::Font(13.0f, juce::Font::bold));
    addAndMakeVisible(*label);
    sliderLabels_.push_back(std::move(label));
}
