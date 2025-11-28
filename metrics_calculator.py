# ==============================================================================
# Evaluation Metrics Implementation
# ==============================================================================

import torch
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import json
from pathlib import Path
from datetime import datetime

class MetricsCalculator:
    """Comprehensive evaluation metrics for image generation"""
    
    def __init__(self, config):
        self.config = config
        
        # Load CLIP for text-image alignment
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(config.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize FID and IS metrics
        self.fid = FrechetInceptionDistance(normalize=True).to(config.device)
        self.inception_score = InceptionScore(normalize=True).to(config.device)
    
    def calculate_clip_score(self, image: Image.Image, text: str) -> float:
        """
        Calculate CLIP score for text-image alignment
        
        Args:
            image: Generated image
            text: Prompt or description
            
        Returns:
            CLIP similarity score
        """
        inputs = self.clip_processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
        
        return score
    
    def calculate_ssim(self, original: Image.Image, edited: Image.Image) -> float:
        """
        Calculate structural similarity between two images
        
        Args:
            original: Original image
            edited: Edited/generated image
            
        Returns:
            SSIM score (0-1, higher is more similar)
        """
        # Convert to numpy arrays
        orig_array = np.array(original.convert('RGB'))
        edit_array = np.array(edited.convert('RGB'))
        
        # Calculate SSIM
        score = ssim(orig_array, edit_array, multichannel=True, channel_axis=2)
        return score
    
    def update_fid(self, real_images: list, fake_images: list):
        """Update FID metric with batch of images"""
        # Convert images to tensors
        real_tensors = [self._image_to_tensor(img) for img in real_images]
        fake_tensors = [self._image_to_tensor(img) for img in fake_images]
        
        real_batch = torch.stack(real_tensors).to(self.config.device)
        fake_batch = torch.stack(fake_tensors).to(self.config.device)
        
        self.fid.update(real_batch, real=True)
        self.fid.update(fake_batch, real=False)
    
    def compute_fid(self) -> float:
        """Compute final FID score"""
        return self.fid.compute().item()
    
    def calculate_inception_score(self, images: list) -> tuple:
        """Calculate Inception Score for image diversity"""
        tensors = [self._image_to_tensor(img) for img in images]
        batch = torch.stack(tensors).to(self.config.device)
        
        self.inception_score.update(batch)
        mean, std = self.inception_score.compute()
        
        return mean.item(), std.item()
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to normalized tensor"""
        img_array = np.array(image.convert('RGB'))
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        return tensor
    
    def export_metrics(self, metrics: dict, save_path: Path):
        """Export metrics to JSON file"""
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics exported to {save_path}")

# Example usage
if __name__ == "__main__":
    from controlnet_pipeline import ProjectConfig
    
    config = ProjectConfig()
    calculator = MetricsCalculator(config)
    
    print("Metrics calculator initialized successfully!")
    print("Available metrics:")
    print("- CLIP Score: Text-image alignment")
    print("- FID: Image quality vs real distribution")
    print("- SSIM: Structure preservation")
    print("- Inception Score: Image diversity")
