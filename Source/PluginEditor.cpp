#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "JuceHeader.h"
#include "BinaryData.h"
#include <cmath>

namespace
{
const auto neonGreen = juce::Colour::fromRGB(0x3C, 0xFF, 0x9B);
const auto deepGrey = juce::Colour::fromRGB(0x16, 0x1E, 0x1F);
const auto darkSlate = juce::Colour::fromRGB(0x0B, 0x12, 0x14);
const auto accentGrey = juce::Colour::fromRGB(0x28, 0x32, 0x34);
const auto textGrey = juce::Colour::fromRGB(0xB0, 0xC8, 0xC3);
constexpr float knobStroke = 4.0f;

std::vector<juce::File> filesFromEnvList(const juce::String& value)
{
    std::vector<juce::File> files;
    for (const auto& token : juce::StringArray::fromTokens(value, ";", ""))
    {
        auto file = juce::File(token.trim().unquoted());
        if (file.existsAsFile())
            files.push_back(file);
    }
    return files;
}

void setFloatParamFromEnv(juce::AudioProcessorValueTreeState& params, const char* envName, const char* paramId)
{
    const auto value = juce::SystemStats::getEnvironmentVariable(envName, {});
    if (value.isEmpty())
        return;

    if (auto* param = dynamic_cast<juce::AudioParameterFloat*>(params.getParameter(paramId)))
        *param = value.getFloatValue();
}

void setIntParamFromEnv(juce::AudioProcessorValueTreeState& params, const char* envName, const char* paramId)
{
    const auto value = juce::SystemStats::getEnvironmentVariable(envName, {});
    if (value.isEmpty())
        return;

    if (auto* param = dynamic_cast<juce::AudioParameterInt*>(params.getParameter(paramId)))
        *param = value.getIntValue();
}

void setChoiceParamFromEnv(juce::AudioProcessorValueTreeState& params, const char* envName, const char* paramId)
{
    const auto value = juce::SystemStats::getEnvironmentVariable(envName, {});
    if (value.isEmpty())
        return;

    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(params.getParameter(paramId)))
        *param = value.getIntValue();
}

juce::Image createNeonPulseImage(const juce::Image& source)
{
    if (!source.isValid())
        return {};

    juce::Image pulse(juce::Image::ARGB, source.getWidth(), source.getHeight(), true);
    juce::Image::BitmapData sourceData(source, juce::Image::BitmapData::readOnly);
    juce::Image::BitmapData pulseData(pulse, juce::Image::BitmapData::writeOnly);

    for (int y = 0; y < source.getHeight(); ++y)
    {
        for (int x = 0; x < source.getWidth(); ++x)
        {
            const auto colour = sourceData.getPixelColour(x, y);
            const float saturation = juce::jlimit(0.0f, 1.0f, (colour.getSaturation() - 0.30f) / 0.50f);
            const float brightness = juce::jlimit(0.0f, 1.0f, (colour.getBrightness() - 0.12f) / 0.55f);
            const float strength = saturation * brightness;

            if (strength > 0.0f)
                pulseData.setPixelColour(x, y, colour.brighter(0.75f).withAlpha(0.85f * strength));
        }
    }

    return pulse;
}
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
    const auto radius = baseSize * 0.40f;
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

            const float renderSize = juce::jmin(baseSize * 1.8f, juce::jmin(static_cast<float>(width), static_cast<float>(height)));
            const float drawX = centre.x - renderSize * 0.5f;
            const float drawY = centre.y - renderSize * 0.5f;
            constexpr int sourceInset = 4;
            const int sourceSize = frameSize - 2 * sourceInset;
            g.drawImage(knobSprite_,
                        static_cast<int>(drawX),
                        static_cast<int>(drawY),
                        static_cast<int>(renderSize),
                        static_cast<int>(renderSize),
                        sourceInset,
                        frameIndex * frameSize + sourceInset,
                        sourceSize,
                        sourceSize);
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
    return juce::Font(juce::FontOptions(14.0f, juce::Font::bold));
}

NeuralMorphingAudioProcessorEditor::NeuralMorphingAudioProcessorEditor(NeuralMorphingAudioProcessor& p)
    : juce::AudioProcessorEditor(&p), processor_(p), tooltipWindow_(nullptr, 500)
{
    setLookAndFeel(&lookAndFeel_);
    formatManager_.registerBasicFormats();
    showStandaloneSource_ = processor_.wrapperType == juce::AudioProcessor::wrapperType_Standalone;

    backgroundImage_ = juce::ImageCache::getFromMemory(BinaryData::vst_background_png, BinaryData::vst_background_pngSize);
    backgroundPulseImage_ = createNeonPulseImage(backgroundImage_);
    logoImage_ = juce::ImageCache::getFromMemory(BinaryData::name_long_logo_png, BinaryData::name_long_logo_pngSize);

    addAndMakeVisible(loadButton_);
    addAndMakeVisible(clearButton_);
    addAndMakeVisible(rebuildButton_);
    addAndMakeVisible(renderButton_);
    addAndMakeVisible(statusLabel_);
    addAndMakeVisible(progressLabel_);
    if (showStandaloneSource_)
    {
        addAndMakeVisible(loadSourceButton_);
        addAndMakeVisible(clearSourceButton_);
        addAndMakeVisible(sourceStatusLabel_);
    }

    setupSlider(temperatureSlider_, "Temperature",
                "Controls palette exploration. Low values are predictable; high values produce more varied, fragmented choices. Usually useful from 0.1 to 1.6.");
    setupSlider(thresholdSlider_, "Threshold",
                "Limits eligible palette matches. Low values are selective; high values are broad. Changes are most audible below about 0.9.");
    setupSlider(continuitySlider_, "Continuity",
                "Moves from scattered grains to sustained trajectories from the same palette sound. High values preserve motion and phrasing.");
    setupSlider(rvqFocusSlider_, "RVQ Focus",
                "Shifts matching from coarse structure toward fine DAC timbral detail. Low is structural; high is brighter and more textural.");
    setupSlider(unitSlider_, "Grain Size",
                "Sets the replacement grain duration. Short grains fragment the sound; long grains preserve gestures and pitch contours.");
    setupSlider(strideSlider_, "Grain Step",
                "Sets the distance between grains. Short steps create dense overlap. Values above Grain Size are clamped to Grain Size.");
    setupSlider(similaritySlider_, "Palette Bias",
                "Controls how closely palette grains resemble the source. High values are compatible and stable; low values are more adventurous.");
    setupSlider(envelopeSlider_, "Envelope",
                "Transfers the source dynamics to the morphed audio. Moderate values add definition; very high values can pump.");
    setupSlider(dryWetSlider_, "Dry/Wet",
                "Linearly mixes source and morphed audio by volume. 0 is source only; 1 is morphed output only.");
    setupSlider(outputSlider_, "Output",
                "Sets final output gain in dB. Large boosts increasingly drive the safety limiter.");

    // Ensure labels are on top by bringing them to front after all sliders are set up
    for (auto& label : sliderLabels_)
    {
        if (label)
            label->toFront(false);
    }
    
    // Set size AFTER creating all components so resized() can position them correctly
    setSize(800, 600);

    temperatureAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "temperature", temperatureSlider_);
    thresholdAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "threshold", thresholdSlider_);
    continuityAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "continuity", continuitySlider_);
    rvqFocusAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "rvqFocus", rvqFocusSlider_);
    unitAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "unit", unitSlider_);
    strideAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "stride", strideSlider_);
    similarityAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "similarity", similaritySlider_);
    envelopeAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "envelopeFollow", envelopeSlider_);
    dryWetAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "dryWet", dryWetSlider_);
    outputAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(processor_.parameters, "outputGain", outputSlider_);

    // Setup backend selector
    addAndMakeVisible(backendSelector_);
    addAndMakeVisible(backendLabel_);
    addAndMakeVisible(processingModeSelector_);
    addAndMakeVisible(processingModeLabel_);
    addAndMakeVisible(bridgeCodecSelector_);
    addAndMakeVisible(bridgeCodecLabel_);
    addAndMakeVisible(swapModeSelector_);
    addAndMakeVisible(swapModeLabel_);
    addAndMakeVisible(statusDisplayLabel_);
    
    backendSelector_.addItem("Native Preferred (+ Bridge Fallback)", 1);
    backendSelector_.addItem("Bridge Only", 2);
    backendSelector_.addItem("Native Only", 3);

    backendAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(processor_.parameters, "backend", backendSelector_);

    processingModeSelector_.addItem("Quality Parity", 1);
    processingModeSelector_.addItem("Live Realtime", 2);
    processingModeAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        processor_.parameters, "processingMode", processingModeSelector_);

    bridgeCodecSelector_.addItem("DAC", 1);
    bridgeCodecSelector_.addItem("SpectroStream", 2);
    bridgeCodecAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(processor_.parameters, "bridgeCodec", bridgeCodecSelector_);

    swapModeSelector_.addItem("Full Layer", 1);
    swapModeSelector_.addItem("RVQ Group", 2);
    swapModeSelector_.addItem("Palette Only", 3);
    swapModeAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(processor_.parameters, "swapMode", swapModeSelector_);

    backendSelector_.setTooltip("Selects native ONNX, the Python bridge, or native with bridge fallback. This changes execution backend, not the parameter mapping.");
    processingModeSelector_.setTooltip("Quality Parity favors coherent matching. Live Realtime uses a faster search and can sound more animated.");
    bridgeCodecSelector_.setTooltip("Selects the bridge codec. DAC is supported by the current bridge; SpectroStream requires a compatible server.");
    swapModeSelector_.setTooltip("Full Layer replaces complete DAC frames, RVQ Group creates a source/palette hybrid, and Palette Only guarantees palette-derived tokens.");
    
    backendLabel_.setVisible(false);
    processingModeLabel_.setVisible(false);
    bridgeCodecLabel_.setVisible(false);
    swapModeLabel_.setVisible(false);
    statusDisplayLabel_.setJustificationType(juce::Justification::centredLeft);
    statusDisplayLabel_.setColour(juce::Label::textColourId, neonGreen.withAlpha(0.8f));

    statusLabel_.setJustificationType(juce::Justification::centredLeft);
    progressLabel_.setJustificationType(juce::Justification::centredRight);
    statusLabel_.setColour(juce::Label::textColourId, textGrey);
    progressLabel_.setColour(juce::Label::textColourId, neonGreen.withAlpha(0.9f));
    sourceStatusLabel_.setJustificationType(juce::Justification::centredLeft);
    sourceStatusLabel_.setColour(juce::Label::textColourId, textGrey);

    loadButton_.setColour(juce::TextButton::buttonColourId, accentGrey);
    clearButton_.setColour(juce::TextButton::buttonColourId, accentGrey);
    rebuildButton_.setColour(juce::TextButton::buttonColourId, accentGrey);
    renderButton_.setColour(juce::TextButton::buttonColourId, accentGrey);
    renderButton_.setButtonText(showStandaloneSource_ ? "Render HQ WAV" : "Arm HQ Render");
    loadSourceButton_.setColour(juce::TextButton::buttonColourId, accentGrey);
    clearSourceButton_.setColour(juce::TextButton::buttonColourId, accentGrey);

    loadButton_.addListener(this);
    clearButton_.addListener(this);
    rebuildButton_.addListener(this);
    renderButton_.addListener(this);
    loadSourceButton_.addListener(this);
    clearSourceButton_.addListener(this);
    backendSelector_.addListener(this);
    processingModeSelector_.addListener(this);
    bridgeCodecSelector_.addListener(this);
    swapModeSelector_.addListener(this);
    temperatureSlider_.addListener(this);
    thresholdSlider_.addListener(this);
    continuitySlider_.addListener(this);
    rvqFocusSlider_.addListener(this);
    unitSlider_.addListener(this);
    strideSlider_.addListener(this);
    similaritySlider_.addListener(this);
    envelopeSlider_.addListener(this);

    startTimerHz(30);
    autoloadStandaloneDemoFilesFromEnvironment();
}

NeuralMorphingAudioProcessorEditor::~NeuralMorphingAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
    loadButton_.removeListener(this);
    clearButton_.removeListener(this);
    rebuildButton_.removeListener(this);
    renderButton_.removeListener(this);
    loadSourceButton_.removeListener(this);
    clearSourceButton_.removeListener(this);
    backendSelector_.removeListener(this);
    processingModeSelector_.removeListener(this);
    bridgeCodecSelector_.removeListener(this);
    swapModeSelector_.removeListener(this);
    temperatureSlider_.removeListener(this);
    thresholdSlider_.removeListener(this);
    continuitySlider_.removeListener(this);
    rvqFocusSlider_.removeListener(this);
    unitSlider_.removeListener(this);
    strideSlider_.removeListener(this);
    similaritySlider_.removeListener(this);
    envelopeSlider_.removeListener(this);
}

void NeuralMorphingAudioProcessorEditor::buildPaletteFromFiles(const std::vector<juce::File>& files)
{
    if (files.empty())
        return;

    lastFiles_ = files;
    lastDirectory_ = files.front().getParentDirectory();

    if (auto* worker = processor_.getPaletteWorker())
    {
        worker->requestBuild(lastFiles_, true,
                             static_cast<int>(unitSlider_.getValue()),
                             static_cast<int>(strideSlider_.getValue()));
        processor_.invalidateMorphCache(true);
    }
}

bool NeuralMorphingAudioProcessorEditor::loadStandaloneSourceFile(const juce::File& file)
{
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager_.createReaderFor(file));
    if (reader == nullptr || reader->lengthInSamples <= 0)
        return false;

    lastDirectory_ = file.getParentDirectory();

    const int64_t length = reader->lengthInSamples;
    juce::AudioBuffer<float> sourceBuffer(static_cast<int>(reader->numChannels), static_cast<int>(length));
    reader->read(&sourceBuffer, 0, static_cast<int>(length), 0, true, true);

    const double desiredSampleRate = processor_.getSampleRate() > 0.0 ? processor_.getSampleRate() : reader->sampleRate;
    if (std::abs(reader->sampleRate - desiredSampleRate) > 1.0)
    {
        const double ratio = reader->sampleRate / desiredSampleRate;
        const int outputSamples = static_cast<int>(std::ceil(static_cast<double>(length) / ratio));
        juce::AudioBuffer<float> resampled(static_cast<int>(reader->numChannels), outputSamples);
        resampled.clear();

        for (int ch = 0; ch < resampled.getNumChannels(); ++ch)
        {
            juce::LagrangeInterpolator interpolator;
            interpolator.reset();
            interpolator.process(ratio,
                                 sourceBuffer.getReadPointer(ch),
                                 resampled.getWritePointer(ch),
                                 outputSamples);
        }

        sourceBuffer = std::move(resampled);
    }

    processor_.setStandaloneSource(std::move(sourceBuffer), desiredSampleRate, file.getFileName());
    return true;
}

void NeuralMorphingAudioProcessorEditor::startStandaloneRenderChooser()
{
    if (renderInProgress_.load(std::memory_order_acquire))
        return;

    if (!processor_.hasStandaloneSource())
    {
        statusLabel_.setText("Load a source sound before rendering.", juce::dontSendNotification);
        return;
    }

    juce::String initialPath = lastDirectory_.exists()
        ? lastDirectory_.getFullPathName()
        : juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getFullPathName();

    auto chooser = std::make_unique<juce::FileChooser>("Render high-quality morph",
                                                       juce::File(initialPath).getChildFile("neural_morphing_hq.wav"),
                                                       "*.wav");
    chooser->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
                         [this, chooserPtr = chooser.get()](const juce::FileChooser& fc)
                         {
                             juce::ignoreUnused(chooserPtr);
                             auto outputFile = fc.getResult();
                             if (outputFile == juce::File{})
                                 return;

                             if (!outputFile.hasFileExtension("wav"))
                                 outputFile = outputFile.withFileExtension(".wav");

                             renderInProgress_.store(true, std::memory_order_release);
                             renderButton_.setEnabled(false);
                             statusLabel_.setText("Rendering HQ WAV...", juce::dontSendNotification);

                             juce::String error;
                             const bool ok = processor_.renderStandaloneSourceToFile(outputFile, error);

                             renderInProgress_.store(false, std::memory_order_release);
                             renderButton_.setEnabled(true);
                             if (ok)
                             {
                                 lastDirectory_ = outputFile.getParentDirectory();
                                 statusLabel_.setText("Rendered: " + outputFile.getFileName(), juce::dontSendNotification);
                             }
                             else
                             {
                                 statusLabel_.setText("Render failed: " + error, juce::dontSendNotification);
                             }
                         });
    chooser.release();
}

void NeuralMorphingAudioProcessorEditor::autoloadStandaloneDemoFilesFromEnvironment()
{
    if (!showStandaloneSource_)
        return;

    applyStandaloneDemoParametersFromEnvironment();

    const auto statusValue = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_DEMO_STATUS_FILE", {});
    if (statusValue.isNotEmpty())
        demoStatusFile_ = juce::File(statusValue.trim().unquoted());

    const auto renderValue = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_DEMO_RENDER_FILE", {});
    if (renderValue.isNotEmpty())
        demoRenderFile_ = juce::File(renderValue.trim().unquoted());

    const auto paletteValue = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_DEMO_PALETTE_FILES", {});
    if (paletteValue.isNotEmpty())
        buildPaletteFromFiles(filesFromEnvList(paletteValue));

    const auto sourceValue = juce::SystemStats::getEnvironmentVariable("NEURAL_MORPHING_DEMO_SOURCE_FILE", {});
    if (sourceValue.isNotEmpty())
        loadStandaloneSourceFile(juce::File(sourceValue.trim().unquoted()));
}

void NeuralMorphingAudioProcessorEditor::applyStandaloneDemoParametersFromEnvironment()
{
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_TEMPERATURE", "temperature");
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_THRESHOLD", "threshold");
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_CONTINUITY", "continuity");
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_RVQ_FOCUS", "rvqFocus");
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_WET_FOCUS", "similarity");
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_PALETTE_BIAS", "similarity");
    setIntParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_GRAIN_SIZE", "unit");
    setIntParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_GRAIN_STEP", "stride");
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_ENVELOPE", "envelopeFollow");
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_DRY_WET", "dryWet");
    setFloatParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_OUTPUT_GAIN", "outputGain");
    setChoiceParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_SWAP_MODE", "swapMode");
    setChoiceParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_PROCESSING_MODE", "processingMode");
    setChoiceParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_BACKEND", "backend");
    setChoiceParamFromEnv(processor_.parameters, "NEURAL_MORPHING_DEMO_BRIDGE_CODEC", "bridgeCodec");
}

void NeuralMorphingAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Draw background first (behind all child components)
    if (backgroundImage_.isValid())
    {
        // Only draw background, don't use stretchToFit to avoid covering UI elements
        g.setOpacity(0.95f);
        g.drawImageWithin(backgroundImage_, 0, 0, getWidth(), getHeight(), juce::RectanglePlacement::stretchToFit);
        g.setOpacity(1.0f);

        if (backgroundPulseImage_.isValid() && backgroundPulse_ > 0.001f)
        {
            g.setOpacity(0.68f * backgroundPulse_);
            g.drawImageWithin(backgroundPulseImage_, 0, 0, getWidth(), getHeight(), juce::RectanglePlacement::stretchToFit);
            g.setOpacity(1.0f);
        }
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
        g.setFont(juce::Font(juce::FontOptions(22.0f, juce::Font::bold)));
        g.drawText("Neural Morphing", logoBounds_, juce::Justification::centredLeft, true);
    }

    if constexpr (NeuralMorphingAudioProcessorEditor::showLayoutDebug_)
    {
        g.setColour(juce::Colours::white.withAlpha(0.65f));
        g.setFont(12.0f);
        g.drawMultiLineText(layoutDebugInfo_, 12, getHeight() - 72, getWidth() - 24);
    }
}

void NeuralMorphingAudioProcessorEditor::paintOverChildren(juce::Graphics& g)
{
    // Paint labels directly on top of all children (including knobs)
    // This is called AFTER all child components are painted
    
    // Custom colors for label boxes
    const auto labelBackgroundColor = juce::Colour::fromRGBA(0x1f, 0x3b, 0x58, 0xff); // #1f3b58ff
    const auto labelOutlineColor = juce::Colour::fromRGBA(0x71, 0xfb, 0xfc, 0xff);    // #71fbfcff
    
    int labelCount = 0;
    int visibleCount = 0;
    int emptyBoundsCount = 0;
    
    for (size_t i = 0; i < sliderLabels_.size(); ++i)
    {
        if (sliderLabels_[i])
        {
            labelCount++;
            
            if (sliderLabels_[i]->isVisible())
            {
                visibleCount++;
                auto bounds = sliderLabels_[i]->getBounds();
                
                // Check if bounds are empty
                if (bounds.isEmpty())
                {
                    emptyBoundsCount++;
                    continue;
                }
                
                // Draw background with custom color #1f3b58ff
                g.setColour(labelBackgroundColor);
                g.fillRoundedRectangle(bounds.toFloat(), 6.0f);
                
                // Draw outline with custom color #71fbfcff
                g.setColour(labelOutlineColor);
                g.drawRoundedRectangle(bounds.toFloat(), 6.0f, 2.0f);
                
                // Draw WHITE text
                g.setColour(juce::Colours::white);
                g.setFont(juce::Font(juce::FontOptions(16.0f, juce::Font::bold)));
                g.drawText(sliderLabels_[i]->getText(), bounds, juce::Justification::centred);
            }
        }
    }
    
    // Debug: show detailed label info
    if constexpr (NeuralMorphingAudioProcessorEditor::showLayoutDebug_)
    {
        g.setColour(juce::Colours::yellow);
        g.setFont(14.0f);
        juce::String info = "Labels: " + juce::String(labelCount) + 
                           " Visible: " + juce::String(visibleCount) + 
                           " EmptyBounds: " + juce::String(emptyBoundsCount) +
                           " Painted: " + juce::String(visibleCount - emptyBoundsCount);
        g.drawText(info, 10, 10, 400, 30, juce::Justification::left);
        
        // Also draw a test rectangle to confirm paintOverChildren is working
        g.setColour(juce::Colours::red.withAlpha(0.5f));
        g.fillRect(10, 45, 100, 20);
        g.setColour(juce::Colours::white);
        g.drawText("TEST PAINT", 10, 45, 100, 20, juce::Justification::centred);
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
    constexpr int buttonCount = 4;
    int usableWidth = buttonSpan.getWidth();
    int perButtonWidth = (usableWidth - (buttonCount - 1) * buttonGap) / buttonCount;
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
    placeButton(renderButton_);
    layoutDebugInfo_ << "Buttons width:" << buttonRow.getWidth()
                     << " per:" << perButtonWidth << '\n';
    currentY += 36 + 10;

    // Status row
    juce::Rectangle<int> statusArea(controlsX, currentY, controlsWidth, 24);
    statusLabel_.setBounds(statusArea.removeFromLeft(statusArea.getWidth() / 2));
    progressLabel_.setBounds(statusArea);
    currentY += 24 + 4;

    if (showStandaloneSource_)
    {
        juce::Rectangle<int> sourceRow(controlsX, currentY, controlsWidth, 32);
        constexpr int sourceGap = 12;
        const int sourceButtonWidth = juce::jmax(140, (sourceRow.getWidth() - sourceGap * 2) / 3);

        auto loadBounds = sourceRow.removeFromLeft(sourceButtonWidth);
        loadSourceButton_.setBounds(loadBounds);
        sourceRow.removeFromLeft(sourceGap);

        auto clearBounds = sourceRow.removeFromLeft(sourceButtonWidth);
        clearSourceButton_.setBounds(clearBounds);
        sourceRow.removeFromLeft(sourceGap);

        sourceStatusLabel_.setBounds(sourceRow);
        currentY += 32 + 4;
    }

    // Backend policy + processing mode row.
    juce::Rectangle<int> backendArea(controlsX, currentY, controlsWidth, 32);
    constexpr int comboGap = 12;
    const int policyWidth = juce::jmax(1, (backendArea.getWidth() - comboGap) / 2);
    backendSelector_.setBounds(backendArea.removeFromLeft(policyWidth).reduced(2));
    backendArea.removeFromLeft(comboGap);
    processingModeSelector_.setBounds(backendArea.reduced(2));
    currentY += 32 + 4;

    // Codec + swap row (side-by-side, intentionally not under backend policy).
    juce::Rectangle<int> codecSwapArea(controlsX, currentY, controlsWidth, 32);
    const int comboWidth = juce::jmax(1, (codecSwapArea.getWidth() - comboGap) / 2);
    bridgeCodecSelector_.setBounds(codecSwapArea.removeFromLeft(comboWidth).reduced(2));
    codecSwapArea.removeFromLeft(comboGap);
    swapModeSelector_.setBounds(codecSwapArea.reduced(2));
    currentY += 32 + 4;

    // Backend status row
    juce::Rectangle<int> backendStatusArea(controlsX, currentY, controlsWidth, 24);
    statusDisplayLabel_.setBounds(backendStatusArea.reduced(2));
    currentY += 24 + 12;

    // Slider grid area
    juce::Rectangle<int> sliderArea(margin, currentY, bounds.getWidth() - 2 * margin, bounds.getBottom() - margin - currentY);
    const int numColumns = 4;
    const int numRows = 3;
    const int sliderWidth = sliderArea.getWidth() / numColumns;
    const int sliderHeight = sliderArea.getHeight() / numRows;

    layoutDebugInfo_ << "slider:" << sliderArea.getWidth() << "x" << sliderArea.getHeight()
                    << " cell:" << sliderWidth << "x" << sliderHeight << '\n';

    juce::Slider* sliders[] = { &temperatureSlider_, &thresholdSlider_, &continuitySlider_, &rvqFocusSlider_,
                                &unitSlider_, &strideSlider_, &similaritySlider_, &envelopeSlider_,
                                &dryWetSlider_, &outputSlider_ };

    int lastKnobSize = 0;

    // First pass: position sliders
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

            constexpr int labelHeight = 32;
            constexpr int verticalGap = 4;
            auto labelBounds = cell.removeFromTop(labelHeight);
            cell.removeFromTop(verticalGap);
            auto knobRegion = cell.reduced(8, 4);

            sliders[index]->setBounds(knobRegion);

            lastKnobSize = juce::jmax(lastKnobSize, juce::jmin(knobRegion.getWidth(), knobRegion.getHeight()));
        }

    // Second pass: position labels AFTER sliders (so they render on top)
    int labelsPositioned = 0;
    for (int row = 0; row < numRows; ++row)
        for (int col = 0; col < numColumns; ++col)
        {
            const int index = row * numColumns + col;
            if (index >= static_cast<int>(std::size(sliders)) || index >= static_cast<int>(sliderLabels_.size()))
                continue;

            juce::Rectangle<int> cell(sliderArea.getX() + col * sliderWidth,
                                      sliderArea.getY() + row * sliderHeight,
                                      sliderWidth,
                                      sliderHeight);

            constexpr int labelHeight = 32;
            auto labelBounds = cell.removeFromTop(labelHeight).reduced(8, 4);

            sliderLabels_[index]->setBounds(labelBounds);
            sliderLabels_[index]->setVisible(true);
            sliderLabels_[index]->toFront(true);
            
            labelsPositioned++;
            
            if constexpr (NeuralMorphingAudioProcessorEditor::showLayoutDebug_)
            {
                layoutDebugInfo_ << "Label[" << index << "]: " << labelBounds.toString() << '\n';
            }
        }

    layoutDebugInfo_ << "Knob region:" << lastKnobSize << "px Labels positioned: " << labelsPositioned;
}

void NeuralMorphingAudioProcessorEditor::buttonClicked(juce::Button* button)
{
    if (button == &loadButton_)
    {
        juce::String initialPath = lastDirectory_.exists() ? lastDirectory_.getFullPathName() : juce::File::getSpecialLocation(juce::File::userHomeDirectory).getFullPathName();
        
        // Use the asynchronous version for better compatibility
        auto chooser = std::make_unique<juce::FileChooser>("Select palette sounds", juce::File(initialPath), "*.wav;*.flac;*.mp3;*.aiff;*.ogg");
        chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::canSelectMultipleItems,
                           [this, chooserPtr = chooser.get()](const juce::FileChooser& fc)
                           {
                               auto results = fc.getResults();
                               if (results.isEmpty())
                                   return;

                               for (const auto& file : results)
                               {
                                   if (file == juce::File{})
                                       continue;

                                   auto alreadyAdded = false;
                                   for (const auto& existing : lastFiles_)
                                   {
                                       if (existing == file)
                                       {
                                           alreadyAdded = true;
                                           break;
                                       }
                                   }

                                   if (!alreadyAdded)
                                       lastFiles_.push_back(file);
                               }

                               if (!lastFiles_.empty())
                                   buildPaletteFromFiles(lastFiles_);
                           });
        chooser.release(); // FileChooser will manage its own lifetime
    }
    else if (button == &clearButton_)
    {
        lastFiles_.clear();
        if (auto* worker = processor_.getPaletteWorker())
        {
            worker->requestBuild({}, true,
                                 static_cast<int>(unitSlider_.getValue()),
                                 static_cast<int>(strideSlider_.getValue()));
            processor_.invalidateMorphCache();
        }
    }
    else if (button == &rebuildButton_)
    {
        if (auto* worker = processor_.getPaletteWorker())
        {
            worker->requestBuild(lastFiles_, true,
                                 static_cast<int>(unitSlider_.getValue()),
                                 static_cast<int>(strideSlider_.getValue()));
            processor_.invalidateMorphCache(true);
        }
    }
    else if (button == &renderButton_)
    {
        if (showStandaloneSource_)
        {
            startStandaloneRenderChooser();
        }
        else
        {
            processor_.armHighQualityRender();
            statusLabel_.setText("HQ render armed. Use DAW freeze or bounce.", juce::dontSendNotification);
        }
    }
    else if (button == &loadSourceButton_)
    {
        juce::String initialPath = lastDirectory_.exists() ? lastDirectory_.getFullPathName() : juce::File::getSpecialLocation(juce::File::userHomeDirectory).getFullPathName();
        auto chooser = std::make_unique<juce::FileChooser>("Select source sound", juce::File(initialPath), "*.wav;*.flac;*.mp3;*.aiff;*.ogg");
        chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                             [this, chooserPtr = chooser.get()](const juce::FileChooser& fc)
                             {
                                 auto result = fc.getResult();
                                 if (result == juce::File{})
                                     return;

                                 loadStandaloneSourceFile(result);
                             });
        chooser.release();
    }
    else if (button == &clearSourceButton_)
    {
        processor_.clearStandaloneSource();
    }
}

void NeuralMorphingAudioProcessorEditor::comboBoxChanged(juce::ComboBox* comboBox)
{
    if (comboBox == &backendSelector_)
    {
        int backendIndex = backendSelector_.getSelectedItemIndex();
        processor_.switchBackend(backendIndex);
    }
    else if (comboBox == &processingModeSelector_)
    {
        int modeIndex = processingModeSelector_.getSelectedItemIndex();
        processor_.setProcessingMode(modeIndex);
    }
    else if (comboBox == &bridgeCodecSelector_)
    {
        int codecIndex = bridgeCodecSelector_.getSelectedItemIndex();
        processor_.setBridgeCodec(codecIndex);
    }
}

void NeuralMorphingAudioProcessorEditor::sliderValueChanged(juce::Slider* slider)
{
    juce::ignoreUnused(slider);
}

void NeuralMorphingAudioProcessorEditor::sliderDragEnded(juce::Slider* slider)
{
    juce::ignoreUnused(slider);
}

void NeuralMorphingAudioProcessorEditor::timerCallback()
{
    const float wetDb = juce::Decibels::gainToDecibels(processor_.visualWetLevel(), -80.0f);
    const float pulseTarget = wetDb > -42.0f ? juce::jlimit(0.0f, 1.0f, (wetDb + 42.0f) / 36.0f) : 0.0f;
    const float smoothing = pulseTarget > backgroundPulse_ ? 0.42f : 0.10f;
    const float previousPulse = backgroundPulse_;
    backgroundPulse_ += smoothing * (pulseTarget - backgroundPulse_);
    if (std::abs(backgroundPulse_ - previousPulse) > 0.001f)
        repaint();

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

    if (!lastFiles_.empty())
        statusText = "Palette: " + juce::String(lastFiles_.size()) + " files | " + statusText;

    if (auto* match = processor_.getMatchWorker())
    {
        if (match->isBusy())
            progressText = "Matching";
    }

    statusLabel_.setText(statusText, juce::dontSendNotification);
    progressLabel_.setText(progressText, juce::dontSendNotification);
    
    // Update backend status
    juce::String backendStatus = processor_.getBackendStatus();

    if (demoRenderFile_ != juce::File{} && !demoRenderAttempted_
        && processor_.isBackendReady() && processor_.hasStandaloneSource()
        && processor_.getPaletteWorker() != nullptr && !processor_.getPaletteWorker()->isBusy())
    {
        demoRenderAttempted_ = true;
        juce::String error;
        demoRenderResult_ = processor_.renderStandaloneSourceToFile(demoRenderFile_, error)
                                ? "ready"
                                : "error:" + error;
    }

    if (demoRenderResult_.isNotEmpty())
        backendStatus += " | render=" + demoRenderResult_;

    statusDisplayLabel_.setText(backendStatus, juce::dontSendNotification);
    if (demoStatusFile_ != juce::File{} && backendStatus != lastDemoStatusText_)
    {
        demoStatusFile_.replaceWithText(backendStatus);
        lastDemoStatusText_ = backendStatus;
    }

    if (showStandaloneSource_)
    {
        juce::String sourceText = "Source: none loaded";
        if (processor_.hasStandaloneSource())
            sourceText = "Source: " + processor_.standaloneSourceName();
        sourceStatusLabel_.setText(sourceText, juce::dontSendNotification);
    }
}

void NeuralMorphingAudioProcessorEditor::setupSlider(juce::Slider& slider,
                                                     const juce::String& name,
                                                     const juce::String& tooltip)
{
    slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 20);
    slider.setName(name);
    slider.setTooltip(tooltip);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::trackColourId, accentGrey);
    addAndMakeVisible(slider);

    // Create a label that will be positioned above the slider
    auto label = std::make_unique<juce::Label>(name + "_label", name);
    label->setJustificationType(juce::Justification::centred);
    label->setColour(juce::Label::textColourId, neonGreen);
    label->setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    label->setColour(juce::Label::outlineColourId, juce::Colours::transparentBlack);
    label->setFont(juce::Font(juce::FontOptions(14.0f, juce::Font::bold)));
    label->setInterceptsMouseClicks(false, false);
    
    // Critical: Add label AFTER slider so it's painted on top
    addAndMakeVisible(*label);
    sliderLabels_.push_back(std::move(label));
}

juce::Rectangle<int> NeuralMorphingAudioProcessorEditor::calculateLogoBounds() const
{
    const int padding = 16;
    const int availableWidth = getWidth() - padding * 2;
    const int availableHeight = getHeight() - padding * 2;
    const int baseHeight = 100;
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
