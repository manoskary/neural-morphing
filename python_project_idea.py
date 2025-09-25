"""
Latent Granular Synthesis with DAC (Descript Audio Codec)
========================================================
creates a "granular codebook" by encoding a source audio corpus 
into latent vector segments, then matches each latent grain of a 
target audio signal to its closest counterpart in the codebook.
"""

import gradio as gr
import librosa, torch
import numpy as np
from tqdm import tqdm
from transformers import DacModel, AutoProcessor
from scipy.spatial.distance import cdist

class LatentGranularSynthesis:
    def __init__(self, model_name="descript/dac_44khz"):
        """Initialize with DAC model from HuggingFace."""
        self.model = DacModel.from_pretrained(model_name, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name, device="auto")
                
        # Get sampling rate from processor
        self.sample_rate = self.processor.sampling_rate
        print(f"Using DAC model: {model_name}")
        print(f"Sample rate: {self.sample_rate} Hz")

        self.unit = 2
        self.stride = 2
        self.temperature = 0.01
        self.threshold = 1.0
        self.files = None
        self.pitch_aug = [-5, -2, 2, 5]
        self.vol_aug = [0.3, 0.7]
    
    def encode(self, audio_array):
        """Encode audio using DAC."""                
        # Process with DAC processor
        inputs = self.processor(
            raw_audio=audio_array, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Encode to get discrete codes and quantized representation
            encoder_outputs = self.model.encode(inputs["input_values"])
            # DAC returns DacEncoderOutput with audio_codes and quantized_representation
            audio_codes = encoder_outputs.audio_codes
            quantized_representation = encoder_outputs.quantized_representation
            
        return audio_codes, quantized_representation
    
    def decode(self, quantized_representation):
        """Decode quantized representation back to audio using DAC."""
        with torch.no_grad():
            # Decode the quantized representation
            audio_values = self.model.decode(quantized_representation)
            
        return audio_values
    
    def set_temperature(self, temperature, threshold):
        self.temperature = temperature * 0.01
        self.threshold = threshold

    def set_unit(self, unit, stride):
        self.unit = unit
        self.stride = stride
        if self.files is not None:
            self.build_dataset(self.files, False)

    def build_dataset(self, files, aug_checkbox: bool):
        self.files = files
        self.codedb = torch.tensor([])
        n_files = 0
        for path in files:
            try:
                y, sr = librosa.load(path, sr=44100)

                # verify y is mono
                if y.ndim > 1:
                    y = librosa.to_mono(y)                

                # Normalize audio
                y = librosa.util.normalize(y)

                if aug_checkbox:
                    # Apply volume augmentation
                    for vol in self.vol_aug:
                        y_vol = y * vol
                        y = np.hstack((y, y_vol))

                    # Apply pitch augmentation
                    for pitch in self.pitch_aug:
                        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
                        y = np.hstack((y, y_pitch))

                # print some info
                print(f"Processing codes for {path}, audio shape: {y.shape}, sample rate: {sr}")                
                # Encode audio
                _, audio_codes = self.encode(y)
                # Use audio codes for granular matching - ensure proper initialization
                if audio_codes is not None:
                    if self.codedb.numel() == 0:
                        self.codedb = audio_codes.cpu()
                    else:
                        self.codedb = torch.cat([self.codedb, audio_codes.cpu()], dim=-1)
                    n_files += 1
            except Exception as e:
                print(e)

        self.db = torch.tensor([])
        for i in range(0, self.codedb.shape[-1], 1):
            code =  self.codedb[:,:,i:i+self.unit]
            if code.shape[-1] != self.unit:
                continue
            self.db = torch.cat((self.db, code), dim=0)
        
        return {"message": f"Done! {n_files} files processed."}
 
    def morph_audio(self, target_file):
        # load target audio
        y, sr = librosa.load(target_file, sr=self.sample_rate, mono=True)
        print(f"Target audio shape: {y.shape}, sample rate: {sr}" )
        print("Creating codes for target audio")
        _, target_codes = self.encode(y)    
        
        if target_codes is None:
            return self.sample_rate, np.zeros(1024, dtype=np.int16)
            
        reconstructed = torch.zeros_like(target_codes).to(target_codes.device)

        # to make it stereo
        if reconstructed.shape[0] == 1:
            reconstructed = torch.vstack([reconstructed, reconstructed]) 

        # find closest code in db
        print("Matching grains...")
        for i in tqdm(range(0, target_codes.shape[-1], self.stride)):
            target_code = target_codes[:,:,i:i+self.unit]
            if target_code.shape[-1] != self.unit:
                continue

            distances = cdist(self.db.reshape(self.db.shape[0], -1).cpu().numpy(), 
                                target_code.reshape(1, -1).cpu().numpy(), 'cosine').squeeze()

            # Apply temperature scaling to logits
            logits = -distances / (self.temperature + 1e-8)
            probabilities = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-8)
            probabilities = np.nan_to_num(probabilities)

            for j in range(2): # to fill stereo buffer
                code_closest = self.db[np.random.choice(self.db.shape[0], p=probabilities/np.sum(probabilities))]
                if min(distances) > self.threshold:
                    code_closest = target_code   
                if i+self.unit < reconstructed.shape[-1]:   
                    reconstructed[j,:,i:i+self.unit] = code_closest
                else:
                    reconstructed[j,:,i:] = code_closest[:,:reconstructed.shape[-1]-i]

        # decode using DAC
        # For DAC, we need to reconstruct the quantized representation
        # This is a simplified approach - in practice, you might need to 
        # properly map the codes back to quantized representation
        print("Decoding reconstructed/morphed audio...")
        y2 = self.decode(reconstructed)  # Using original quantized as fallback
        
        # Handle DAC decoder output - check the actual structure
        
        audio_output = y2.audio_values
            
        # Ensure we have a valid tensor and extract the tensor data
        if audio_output is None:
            return sr, np.zeros(1024, dtype=np.int16)
        
        # Handle DacDecoderOutput - extract the actual tensor
        if hasattr(audio_output, 'squeeze'):
            final_audio = audio_output
        elif hasattr(audio_output, 'data'):
            final_audio = audio_output.data
        elif hasattr(audio_output, 'tensor'):
            final_audio = audio_output.tensor
        else:
            # Try to convert to tensor if it's not already
            final_audio = torch.as_tensor(audio_output)
            
        sr = self.sample_rate
        return sr, (final_audio.cpu().numpy().squeeze().transpose() * 31000).astype(np.int16)


synth = LatentGranularSynthesis()

def build_dataset(files, aug_checkbox):
    return synth.build_dataset(files, aug_checkbox)

def morph_audio(target_file):
    return synth.morph_audio(target_file)

def temperature(temperature, threshold):
    return synth.set_temperature(temperature, threshold)

def unit(unit, stride):
    return synth.set_unit(unit, stride)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # gr.Label("Upload your audio files to train a model")
            db_file = gr.File(file_count="multiple", label="Source Sounds")
            aug_checkbox = gr.Checkbox(label="Apply Augmentation")
            b1 = gr.Button("Process source sounds")
            text = gr.Textbox(label="Result")

        with gr.Column():
            # gr.Label("Upload a target audio file to morph")
            target_file = gr.File(label="Target sound")
            
            # Parameter controls
            with gr.Row():
                temp_slider = gr.Slider(0.1, 2.0, value=1.0, label="Temperature")
                threshold_slider = gr.Slider(0.1, 2.0, value=1.0, label="Threshold")
            with gr.Row():
                unit_slider = gr.Slider(1, 10, value=2, step=1, label="Unit Size")
                stride_slider = gr.Slider(1, 10, value=2, step=1, label="Stride")
                
            b2 = gr.Button("Morph Audio")
            audioplayer = gr.Audio(label="Output")

    # Connect the interface elements
    temp_slider.change(temperature, inputs=[temp_slider, threshold_slider])
    threshold_slider.change(temperature, inputs=[temp_slider, threshold_slider])
    unit_slider.change(unit, inputs=[unit_slider, stride_slider])
    stride_slider.change(unit, inputs=[unit_slider, stride_slider])
    
    b1.click(build_dataset, inputs=[db_file, aug_checkbox], outputs=text)
    b2.click(morph_audio, inputs=target_file, outputs=audioplayer)

demo.launch()
