Model Name / Tool,Type,Primary Use in Your Project//
Stable Diffusion v1.5,Diffusion Model,The core model used to generate the initial images from the text prompts (BASE_MODEL_ID). It creates the primary visual content for the lessons.
LoRA Adapter (Local),Fine-Tuning Weights,Your custom weights (./biology_lora_adapter) applied on top of Stable Diffusion to make the generated images culturally relevant to African biology education.
ControlNet (Canny),Conditional Model,"Used to impose structural control on the image generation. It takes a Canny Edge Map (a line drawing of the initial image) and forces Stable Diffusion to follow that outline, improving image coherence."
CLIP (ViT-B/32),Vision-Language Model,"Used for Cultural Relevance Validation. It measures the similarity (coherence score) between the generated Image and the descriptive Text Prompt, helping ensure the visual content matches the intended topic."
SAM (Segment Anything Model),Segmentation Model,"Used for Image Refinement. If the CLIP score is low, SAM identifies and masks the primary biological objects in the image to help focus the visual content, effectively cleaning up the background or less relevant parts."
gTTS (Google Text-to-Speech),Speech Synthesis Tool,"Used to generate the audio narration for the lessons in English, Luganda, and Swahili from the text scripts in your CSV file."
Gradio,UI/UX Framework,"Used to build the interactive user interface (the chatbot) that allows a user to input a topic and view the generated lesson steps, images, and audio files.".
