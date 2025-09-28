#pragma once

#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

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
    void resized() override;

private:
    void buttonClicked(juce::Button*) override;
    void comboBoxChanged(juce::ComboBox*) override;
    void timerCallback() override;
    void setupSlider(juce::Slider& slider, const juce::String& name);

    NeuralMorphingAudioProcessor& processor_;

    juce::TextButton loadButton_{ "Add Target Files" };
    juce::TextButton clearButton_{ "Clear Palette" };
    juce::TextButton rebuildButton_{ "Rebuild" };

    juce::Label statusLabel_;
    juce::Label progressLabel_;

    juce::Slider temperatureSlider_;
    juce::Slider thresholdSlider_;
    juce::Slider unitSlider_;
    juce::Slider strideSlider_;
    juce::Slider similaritySlider_;
    juce::Slider envelopeSlider_;
    juce::Slider dryWetSlider_;
    juce::Slider outputSlider_;

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralMorphingAudioProcessorEditor)
};
