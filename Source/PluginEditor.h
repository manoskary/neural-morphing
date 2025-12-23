#pragma once

#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

class NeuralMorphingLookAndFeel : public juce::LookAndFeel_V4
{
public:
    NeuralMorphingLookAndFeel();

    void drawRotarySlider(juce::Graphics&, int x, int y, int width, int height,
                          float sliderPosProportional, float rotaryStartAngle,
                          float rotaryEndAngle, juce::Slider&) override;
    void drawButtonBackground(juce::Graphics&, juce::Button&, const juce::Colour& backgroundColour,
                              bool isMouseOverButton, bool isButtonDown) override;
    void drawComboBox(juce::Graphics&, int width, int height, bool isButtonDown,
                      int buttonX, int buttonY, int buttonW, int buttonH, juce::ComboBox&) override;
    int getComboBoxBorderThickness(juce::ComboBox&) {
        return 0;
    }
    juce::Font getComboBoxFont(juce::ComboBox&) override;

private:
    juce::Image knobSprite_;
};

class NeuralMorphingAudioProcessor;

class NeuralMorphingAudioProcessorEditor : public juce::AudioProcessorEditor,
                                           private juce::Button::Listener,
                                           private juce::ComboBox::Listener,
                                           private juce::Timer
{
public:
    explicit NeuralMorphingAudioProcessorEditor(NeuralMorphingAudioProcessor&);
    ~NeuralMorphingAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void paintOverChildren(juce::Graphics&) override;
    void resized() override;

private:
    void buttonClicked(juce::Button*) override;
    void comboBoxChanged(juce::ComboBox*) override;
    void timerCallback() override;
    void setupSlider(juce::Slider& slider, const juce::String& name);
    juce::Rectangle<int> calculateLogoBounds() const;

    NeuralMorphingAudioProcessor& processor_;

    NeuralMorphingLookAndFeel lookAndFeel_;

    static constexpr bool showLayoutDebug_ = false;

    juce::TextButton loadButton_{ "Add Target Files" };
    juce::TextButton clearButton_{ "Clear Palette" };
    juce::TextButton rebuildButton_{ "Rebuild" };
    juce::TextButton loadSourceButton_{ "Load Source Audio" };
    juce::TextButton clearSourceButton_{ "Clear Source" };

    juce::Label statusLabel_;
    juce::Label progressLabel_;
    juce::Label sourceStatusLabel_;

    juce::Slider temperatureSlider_;
    juce::Slider thresholdSlider_;
    juce::Slider unitSlider_;
    juce::Slider strideSlider_;
    juce::Slider similaritySlider_;
    juce::Slider envelopeSlider_;
    juce::Slider dryWetSlider_;
    juce::Slider outputSlider_;

    std::vector<std::unique_ptr<juce::Label>> sliderLabels_;
    juce::String layoutDebugInfo_;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> temperatureAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> thresholdAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> unitAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> strideAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> similarityAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> envelopeAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> dryWetAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> outputAttachment_;

    // Backend selection UI
    juce::ComboBox backendSelector_;
    juce::Label backendLabel_{ "Backend", "Backend:" };
    juce::Label statusDisplayLabel_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> backendAttachment_;

    juce::File lastDirectory_;
    std::vector<juce::File> lastFiles_;
    bool showStandaloneSource_ = false;
    juce::AudioFormatManager formatManager_;

    juce::Image backgroundImage_;
    juce::Image logoImage_;
    mutable juce::Rectangle<int> logoBounds_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralMorphingAudioProcessorEditor)
};
