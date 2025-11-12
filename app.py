import torch
from diffusers import StableDiffusionPipeline, ControlNetModel
from gtts import gTTS
import gradio as gr
import pandas as pd
import os
from PIL import Image
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import clip
from huggingface_hub import hf_hub_download
from cachetools import TTLCache, cached
import json
from pathlib import Path

# --- CONFIGURATION ---
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_ADAPTER_PATH = "./biology_lora_adapter" 
DATA_FILE = "O level Biology.csv"
MODEL_CACHE_DIR = "./models"
CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Creating model cache directory
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def download_sam_checkpoint():
    """Download SAM checkpoint if not exists"""
    sam_path = os.path.join(MODEL_CACHE_DIR, "sam_vit_h_4b8939.pth")
    if not os.path.exists(sam_path):
        print("Downloading SAM checkpoint...")
        try:
            import urllib.request
            urllib.request.urlretrieve(SAM_CHECKPOINT_URL, sam_path)
            print("SAM checkpoint downloaded successfully!")
        except Exception as e:
            print(f"Failed to download SAM checkpoint: {e}")
            return None
    return sam_path

# Loading the dataset globally
try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    raise RuntimeError(f"Error: Could not find '{DATA_FILE}'. Please check the file name and location.")

# --- MODEL INITIALIZATION ---
class EnhancedPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        if self.device == "cpu":
            print("Warning: CUDA GPU not detected. Running on CPU may be extremely slow.")
        
        self.pipe = self._init_stable_diffusion()
        self.controlnet = self._init_controlnet()
        self.clip_model = self._init_clip()
        self.sam_generator = self._init_sam_automatic() 
        
        print("--- Enhanced Pipeline Loaded Successfully! ---")

    def _init_stable_diffusion(self):
        print("Loading Stable Diffusion...")
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=self.dtype,
            safety_checker=None
        )
        try:
            pipe.load_lora_weights(LORA_ADAPTER_PATH)
            print("LoRA weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load LoRA weights: {e}")
        
        pipe.to(self.device)
        return pipe

    def _init_controlnet(self):
        print("Loading ControlNet...")
        try:
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_ID,
                torch_dtype=self.dtype,
                cache_dir=MODEL_CACHE_DIR
            ).to(self.device)
            print("ControlNet loaded successfully!")
            return controlnet
        except Exception as e:
            print(f"Error loading ControlNet: {e}")
            raise 

    def _init_clip(self):
        print("Loading CLIP...")
        try:
            model, preprocess = clip.load("ViT-B/32", device=self.device, download_root=MODEL_CACHE_DIR)
            print("CLIP loaded successfully!")
            return {"model": model, "preprocess": preprocess}
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            raise

    def _init_sam_automatic(self):
        print("Loading SAM Automatic Mask Generator...")
        sam_path = download_sam_checkpoint()
        if not sam_path:
            raise RuntimeError("Failed to initialize SAM. Checkpoint download failed.")
            
        try:
            sam = sam_model_registry["vit_h"](checkpoint=sam_path)
            sam.to(device=self.device)
            mask_generator = SamAutomaticMaskGenerator(sam)
            print("SAM Automatic Mask Generator loaded successfully!")
            return mask_generator
        except Exception as e:
            print(f"Error loading SAM: {e}")
            raise

    def generate_image(self, prompt, image_prompt):
        full_prompt = f"{prompt}, {image_prompt}"
        negative_prompt = "ugly, deformed, low quality, abstract, blurry, text"
        
        generator = torch.manual_seed(42)
        init_image = np.array(self.pipe(full_prompt, num_inference_steps=1, generator=generator, negative_prompt=negative_prompt).images[0])
        canny_image = cv2.Canny(init_image, 100, 200)
        canny_image = Image.fromarray(canny_image)

        image = self.pipe(
            full_prompt,
            image=canny_image,
            controlnet=self.controlnet,
            num_inference_steps=50,
            guidance_scale=8.0,
            generator=generator,
            negative_prompt=negative_prompt
        ).images[0]

        return image

    def validate_cultural_relevance(self, image, prompt):
        image_input = self.clip_model["preprocess"](image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model["model"].encode_image(image_input)
            text_features = self.clip_model["model"].encode_text(text_input)
            similarity = torch.cosine_similarity(image_features, text_features)

        return similarity.item()

    def refine_with_sam(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        masks_list = self.sam_generator.generate(image)
        
        if not masks_list:
            print("SAM found no segments to refine.")
            return Image.fromarray(image) 

        combined_mask = np.zeros(masks_list[0]['segmentation'].shape, dtype=bool)
        for mask_data in masks_list:
            combined_mask = np.logical_or(combined_mask, mask_data['segmentation'])
        
        refined_image = np.zeros_like(image)
        refined_image[combined_mask] = image[combined_mask]
        
        return Image.fromarray(refined_image)

# Initializing the pipeline
pipeline = None
try:
    pipeline = EnhancedPipeline()
except Exception as e:
    print(f"CRITICAL STARTUP FAILURE: {e}")

# --- GENERATION LOGIC  ---
def generate_full_lesson(topic, level):
    
    if pipeline is None:
        raise gr.Error("Pipeline failed to initialize. Check Colab logs for startup errors.")
    
    # 1.Find ALL matching rows for the topic
    topic_data_df = data[(data['Topic'].str.strip().str.lower() == topic.strip().lower()) & 
                         (data['Level'] == level)]
    
    if topic_data_df.empty:
        raise gr.Error("Topic not found. Please enter the exact topic name (e.g., 'Cells') and check the Level.")
    
    lesson_sequence = []
    lesson_sequence.append((None, f"## 📚 Lesson: {topic} ({level})"))
    
    # 2. Loop through each row and generate content
    for index, row in topic_data_df.iterrows():
        print(f"\n--- Generating Step {index+1}/{len(topic_data_df)}: {row['Instructional Step']} ---")
        
        # Get data from CSV row
        image_prompt = row['Visual Prompt']
        narration_script = row['Narration Script']
        step_title = row['Instructional Step']
        
        #  Get Luganda and Swahili narrations ---
        narration_lg = row['Luganda_Narration']
        narration_sw = row['Swahili_Narration']

        # 3. Generate and Validate Image
        image = pipeline.generate_image(topic, image_prompt)
        
        relevance_score = pipeline.validate_cultural_relevance(image, image_prompt)
        print(f"Cultural relevance score: {relevance_score}")
        
        if relevance_score < 0.25: 
            print("Low cultural relevance - applying refinement")
            image = pipeline.refine_with_sam(image)
        
        #  Saving image to a temporary file ---
        image_path = f"image_step_{index}.png"
        image.save(image_path)

        # Generate all three Audio files
        tts_en = gTTS(text=narration_script, lang='en')
        audio_path_en = f"audio_step_{index}_en.mp3"
        tts_en.save(audio_path_en)
        
        tts_lg = gTTS(text=narration_lg, lang='en') 
        audio_path_lg = f"audio_step_{index}_lg.mp3"
        tts_lg.save(audio_path_lg)

        tts_sw = gTTS(text=narration_sw, lang='sw')
        audio_path_sw = f"audio_step_{index}_sw.mp3"
        tts_sw.save(audio_path_sw)
        
        # Add this step to the chatbot history
        lesson_sequence.append((None, f"### {step_title}"))
        
        #  Add image file path to chat ---
        lesson_sequence.append((None, (image_path,))) # Pass image path as a tuple
        
        lesson_sequence.append((None, narration_script)) # Display text
        
        # --- FIX 2: Add all three audio files to chat ---
        lesson_sequence.append((None, (audio_path_en,))) # English
        lesson_sequence.append((None, (audio_path_lg,))) # Luganda
        lesson_sequence.append((None, (audio_path_sw,))) # Swahili
        
        lesson_sequence.append((None, "---")) # Add a separator

    print("\n--- Full Lesson Generation Complete! ---")
    
    # Return the complete chat history
    return lesson_sequence

# --- GRADIO INTERFACE ---
iface = gr.Interface(
    fn=generate_full_lesson,
    inputs=[
        gr.Textbox(label="Enter Biology Topic (e.g., 'Cells')", value="Introduction to Biology"),
        gr.Dropdown(label="Select Academic Level", choices=data['Level'].unique().tolist(), value="Senior 1")
    ],
    outputs=[
        gr.Chatbot(label="Full Lesson Sequence", height=800)
    ],
    title="🔬 Final Project: Full Lesson Generator",
    description="Enter a topic (like 'Cells' or 'Introduction to Biology') to generate the complete, multi-step visual and audio lesson."
)

if __name__ == "__main__":
    # The 'server_name="0.0.0.0"' ensures the app is accessible outside the container
    iface.launch(server_name="0.0.0.0", server_port=7860)