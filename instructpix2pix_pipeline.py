# ==============================================================================
# InstructPix2Pix Pipeline Implementation
# ==============================================================================

from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image

class InstructPix2PixPipeline:
    """Implements instruction-based image editing"""
    
    def __init__(self, config):
        self.config = config
        
        # Load InstructPix2Pix pipeline
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=config.dtype
        ).to(config.device)
    
    def edit_image(self, 
                   image: Image.Image, 
                   instruction: str,
                   num_inference_steps: int = 20,
                   text_guidance_scale: float = 7.5,
                   image_guidance_scale: float = 1.5) -> Image.Image:
        """
        Edit image based on text instruction
        
        Args:
            image: Input image to edit
            instruction: Natural language editing instruction
            num_inference_steps: Number of diffusion steps
            text_guidance_scale: Strength of text guidance
            image_guidance_scale: Strength of image preservation
            
        Returns:
            Edited image
        """
        result = self.pipeline(
            prompt=instruction,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale
        ).images[0]
        
        return result

# Example usage
if __name__ == "__main__":
    from controlnet_pipeline import ProjectConfig
    
    config = ProjectConfig()
    pipeline = InstructPix2PixPipeline(config)
    
    # Example instructions
    instructions = [
        "Change hair color to blonde",
        "Add dramatic blue lighting from the left",
        "Make the lighting warm like sunset",
        "Change to a gentle smile"
    ]
    
    print("InstructPix2Pix pipeline initialized successfully!")
    print(f"Example instructions: {instructions}")
