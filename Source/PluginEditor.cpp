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
                                         static_cast<float>(width), static_cast<float>(height)).reduced(2.0f);

    const auto baseSize = juce::jmin(bounds.getWidth(), bounds.getHeight());
    const auto radius = baseSize * 0.45f;
    const auto centre = bounds.getCentre();

    if (knobSprite_.isValid())
    {
        const int frameSize = knobSprite_.getWidth();
        const int frameCount = frameSize > 0 ? knobSprite_.getHeight() / frameSize : 0;

        if (frameSize > 0 && frameCount > 0)
        {
            const float sliderNorm = juce::jlimit(0.0f, 1.0f, sliderPosProportional);
            const int frameIndex = juce::jlimit(0, frameCount - 1,
                                                static_cast<int>(std::round(sliderNorm * (frameCount - 1))));

            const float renderSize = juce::jmin(baseSize * 2.4f, juce::jmin(static_cast<float>(width), static_cast<float>(height)));
            const float drawX = centre.x - renderSize * 0.5f;
            const float drawY = centre.y - renderSize * 0.5f;
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
    setSize(800, 600);

    backgroundImage_ = juce::ImageCache::getFromMemory(BinaryData::vst_background_png, BinaryData::vst_background_pngSize);
    logoImage_ = juce::ImageCache::getFromMemory(BinaryData::name_long_logo_png, BinaryData::name_long_logo_pngSize);

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
    if (backgroundImage_.isValid())
    {
        g.drawImageWithin(backgroundImage_, 0, 0, getWidth(), getHeight(), juce::RectanglePlacement::stretchToFit);
    }
    else
    {
        juce::ColourGradient gradient(darkSlate, 0.0f, 0.0f,
                                      deepGrey.darker(0.4f), 0.0f, static_cast<float>(getHeight()), false);
        gradient.addColour(0.5f, deepGrey);
        g.setGradientFill(gradient);
        g.fillAll();
    }

    logoBounds_ = calculateLogoBounds();
    if (logoImage_.isValid())
    {
        g.drawImageWithin(logoImage_, logoBounds_.getX(), logoBounds_.getY(),
                          logoBounds_.getWidth(), logoBounds_.getHeight(), juce::RectanglePlacement::centred);
    }
    else
    {
        g.setColour(neonGreen);
        g.setFont(juce::Font(22.0f, juce::Font::bold));
        g.drawText("Neural Morphing", logoBounds_, juce::Justification::centredLeft, true);
    }

    if constexpr (NeuralMorphingAudioProcessorEditor::showLayoutDebug_)
    {
        g.setColour(juce::Colours::white.withAlpha(0.65f));
        g.setFont(12.0f);
        g.drawMultiLineText(layoutDebugInfo_, 12, getHeight() - 72, getWidth() - 24);
    }
}

void NeuralMorphingAudioProcessorEditor::resized()
{
    logoBounds_ = calculateLogoBounds();

    layoutDebugInfo_.clear();

    const int margin = 16;
    auto bounds = getLocalBounds();
    int availableWidth = bounds.getWidth() - 2 * margin;
    int currentY = std::max(bounds.getY() + margin, logoBounds_.getBottom() + margin);

    const int controlsX = margin;
    const int controlsWidth = availableWidth;

    // Button row positioned beneath the logo.
    juce::Rectangle<int> buttonRow(controlsX, currentY, controlsWidth, 36);
    auto buttonSpan = buttonRow;
    constexpr int buttonGap = 12;
    int usableWidth = buttonSpan.getWidth();
    int perButtonWidth = (usableWidth - 2 * buttonGap) / 3;
    perButtonWidth = juce::jmax(120, perButtonWidth);
    perButtonWidth = juce::jmin(perButtonWidth, usableWidth);

    auto placeButton = [&](juce::TextButton& button)
    {
        auto bounds = buttonSpan.removeFromLeft(perButtonWidth);
        button.setBounds(bounds);
        buttonSpan.removeFromLeft(buttonGap);
    };

    placeButton(loadButton_);
    placeButton(clearButton_);
    placeButton(rebuildButton_);
    layoutDebugInfo_ << "Buttons width:" << buttonRow.getWidth()
                     << " per:" << perButtonWidth << '\n';
    currentY += 36 + 10;

    // Status row
    juce::Rectangle<int> statusArea(controlsX, currentY, controlsWidth, 24);
    statusLabel_.setBounds(statusArea.removeFromLeft(statusArea.getWidth() / 2));
    progressLabel_.setBounds(statusArea);
    currentY += 24 + 4;

    // Backend selector row
    juce::Rectangle<int> backendArea(controlsX, currentY, controlsWidth, 32);
    backendSelector_.setBounds(backendArea.removeFromLeft(220).reduced(2));
    currentY += 32 + 4;

    // Backend status row
    juce::Rectangle<int> backendStatusArea(controlsX, currentY, controlsWidth, 24);
    statusDisplayLabel_.setBounds(backendStatusArea.reduced(2));
    currentY += 24 + 12;

    // Slider grid area
    juce::Rectangle<int> sliderArea(margin, currentY, bounds.getWidth() - 2 * margin, bounds.getBottom() - margin - currentY);
    const int numColumns = 4;
    const int numRows = 2;
    const int sliderWidth = sliderArea.getWidth() / numColumns;
    const int sliderHeight = sliderArea.getHeight() / numRows;

    layoutDebugInfo_ << "slider:" << sliderArea.getWidth() << "x" << sliderArea.getHeight()
                    << " cell:" << sliderWidth << "x" << sliderHeight << '\n';

    juce::Slider* sliders[] = { &temperatureSlider_, &thresholdSlider_, &unitSlider_, &strideSlider_,
                                &similaritySlider_, &envelopeSlider_, &dryWetSlider_, &outputSlider_ };

    int lastKnobSize = 0;

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

            constexpr int labelHeight = 28;
            auto labelBounds = cell.removeFromTop(labelHeight).reduced(4, 2);
            auto knobRegion = cell.reduced(4, 2);

            sliders[index]->setBounds(knobRegion);
            if (index < static_cast<int>(sliderLabels_.size()))
            {
                sliderLabels_[index]->setBounds(labelBounds);
                sliderLabels_[index]->setVisible(true);
                sliderLabels_[index]->toFront(false);
            }

            lastKnobSize = juce::jmax(lastKnobSize, juce::jmin(knobRegion.getWidth(), knobRegion.getHeight()));
        }

    layoutDebugInfo_ << "Knob region:" << lastKnobSize << "px";
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
    label->setColour(juce::Label::textColourId, juce::Colours::white);
    label->setColour(juce::Label::backgroundColourId, neonGreen.withAlpha(0.18f));
    label->setColour(juce::Label::outlineColourId, neonGreen.withAlpha(0.35f));
    label->setOpaque(true);
    label->setFont(juce::Font(14.0f, juce::Font::bold));
    label->setInterceptsMouseClicks(false, false);
    addAndMakeVisible(*label);
    sliderLabels_.push_back(std::move(label));
}

juce::Rectangle<int> NeuralMorphingAudioProcessorEditor::calculateLogoBounds() const
{
    const int padding = 16;
    const int availableWidth = getWidth() - padding * 2;
    const int availableHeight = getHeight() - padding * 2;
    const int baseHeight = 80;
    const int desiredHeight = juce::jmin(baseHeight * 2, availableHeight);

    if (logoImage_.isValid())
    {
        auto imageAspect = static_cast<float>(logoImage_.getWidth()) / static_cast<float>(logoImage_.getHeight());
        int targetHeight = juce::jmin(desiredHeight, static_cast<int>(availableWidth / imageAspect));
        if (targetHeight <= 0)
            targetHeight = desiredHeight;
        int targetWidth = static_cast<int>(imageAspect * targetHeight);
        targetWidth = juce::jlimit(0, availableWidth, targetWidth);
        targetHeight = juce::jlimit(0, availableHeight, targetHeight);

        int x = (getWidth() - targetWidth) / 2;
        int y = padding;

        return { x, y, targetWidth, targetHeight > 0 ? targetHeight : desiredHeight };
    }

    int fallbackWidth = juce::jmin(availableWidth, baseHeight * 4);
    int fallbackHeight = baseHeight;
    int x = (getWidth() - fallbackWidth) / 2;
    return { x, padding, fallbackWidth, fallbackHeight };
}
