# ==============================================================================
# ControlNet Pipeline Implementation
# ==============================================================================

from dataclasses import dataclass
from pathlib import Path
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from PIL import Image

@dataclass
class ProjectConfig:
    """Centralized configuration management"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    image_resolution: int = 512
    controlnet_model: str = "lllyasviel/sd-controlnet-openpose"
    base_model: str = "runwayml/stable-diffusion-v1-5"

class ControlNetPipeline:
    """Implements pose-based image generation using ControlNet"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_model,
            torch_dtype=config.dtype
        )
        
        # Load pipeline
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            config.base_model,
            controlnet=controlnet,
            torch_dtype=config.dtype
        ).to(config.device)
    
    def extract_pose(self, image: Image.Image) -> Image.Image:
        """Extract pose skeleton from image"""
        return self.pose_detector(image)
    
    def generate(self, prompt: str, pose_image: Image.Image, 
                 num_inference_steps: int = 30, 
                 guidance_scale: float = 7.5) -> Image.Image:
        """Generate image with pose control"""
        skeleton = self.extract_pose(pose_image)
        
        result = self.pipeline(
            prompt=prompt,
            image=skeleton,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        return result

if __name__ == "__main__":
    config = ProjectConfig()
    pipeline = ControlNetPipeline(config)
    print("ControlNet pipeline initialized successfully!")
