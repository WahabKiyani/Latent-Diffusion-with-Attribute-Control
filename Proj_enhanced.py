# ==============================================================================
# FINAL PROJECT: Latent Diffusion with Attribute Control
# Course: Generative AI, SPRING 2025
# Instructor: Dr. Akhtar Jamil
# ==============================================================================
# Student Names: Muhammad Abdul Wahab Kiyani (22i-1178), Syed Ahmed Ali Zaidi (22i-1237)
# Project Type: Comparative Analysis - Architectural vs. Instruction-Based Control
# ==============================================================================

"""
This implementation compares two approaches for attribute control in latent diffusion:
1. Architectural Control: ControlNet for pose/structure manipulation
2. Instruction-Based Control: InstructPix2Pix for semantic attribute editing

The project follows modern MLOps practices with modular architecture, comprehensive
evaluation metrics, and Docker deployment support.
"""

import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime

# Diffusion Model Libraries
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    StableDiffusionInstructPix2PixPipeline
)
from controlnet_aux import OpenposeDetector

# Evaluation Metrics
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torch.nn.functional as F

# ==============================================================================
# CONFIGURATION CLASS (Modern Industry Standard)
# ==============================================================================

@dataclass
class ProjectConfig:
    """Centralized configuration management for all model parameters"""
    
    # System Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Model Paths
    checkpoint_path: str = "pretrained/Realistic_Vision_V4.0.safetensors"
    controlnet_model: str = "lllyasviel/sd-controlnet-openpose"
    instruct_pix2pix_model: str = "timbrooks/instruct-pix2pix"
    
    # Generation Parameters
    default_steps: int = 30
    default_guidance: float = 7.5
    image_resolution: int = 512
    
    # Evaluation Settings
    clip_model: str = "openai/clip-vit-base-patch32"
    num_fid_samples: int = 50
    
    # Output Directories
    output_dir: Path = Path("outputs")
    results_dir: Path = Path("outputs/results")
    metrics_dir: Path = Path("outputs/metrics")
    visualizations_dir: Path = Path("outputs/visualizations")
    
    def __post_init__(self):
        """Create necessary directories on initialization"""
        for directory in [self.output_dir, self.results_dir, 
                         self.metrics_dir, self.visualizations_dir]:
            directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATA LOADING AND PREPROCESSING (Rubric Item 2: 5 marks)
# ==============================================================================

class DatasetManager:
    """
    Handles dataset loading, preprocessing, and visualization.
    Supports both local images and test datasets for evaluation.
    """
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.sample_images: List[Image.Image] = []
        self.metadata: Dict = {}
        
    def load_sample_dataset(self, dataset_path: Optional[str] = None) -> List[Image.Image]:
        """
        Load sample images for testing and evaluation.
        
        Args:
            dataset_path: Path to image directory (optional)
            
        Returns:
            List of PIL Images
        """
        if dataset_path and os.path.exists(dataset_path):
            image_files = [f for f in os.listdir(dataset_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.sample_images = [
                Image.open(os.path.join(dataset_path, f)).convert('RGB') 
                for f in image_files[:20]  # Limit to 20 samples
            ]
        else:
            # Generate placeholder message if no dataset provided
            print("ℹ️ No custom dataset provided. Using uploaded images for evaluation.")
            
        return self.sample_images
    
    def preprocess_image(self, image: Image.Image, target_size: int = 512) -> Image.Image:
        """
        Standardize image preprocessing pipeline.
        
        Args:
            image: Input PIL Image
            target_size: Target resolution for model input
            
        Returns:
            Preprocessed PIL Image
        """
        # Resize maintaining aspect ratio
        w, h = image.size
        if w > h:
            new_w, new_h = target_size, int(target_size * h / w)
        else:
            new_w, new_h = int(target_size * w / h), target_size
            
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop to square
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))
        
        return image
    
    def visualize_dataset_statistics(self, save_path: Path):
        """
        Generate and save dataset visualization (required for documentation).
        
        Args:
            save_path: Path to save visualization
        """
        if not self.sample_images:
            return
            
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Sample Dataset Visualization', fontsize=16)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(self.sample_images):
                ax.imshow(self.sample_images[idx])
                ax.axis('off')
                ax.set_title(f'Sample {idx+1}')
            else:
                ax.axis('off')
                
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Dataset visualization saved to {save_path}")

# ==============================================================================
# SUPER-RESOLUTION & QUALITY ENHANCEMENT (Proposal Phase 1)
# ==============================================================================

class SuperResolutionEnhancer:
    """
    Implements 512x512 → 1024x1024 upscaling using Real-ESRGAN.
    Addresses Proposal Requirement: Quality Enhancement - Upscaling
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.model_loaded = False
        
    def load_model(self):
        """Lazy loading of Real-ESRGAN model"""
        try:
            # Compatibility patch for newer torchvision versions
            import sys
            import torchvision.transforms.functional as F
            if not hasattr(sys.modules.get('torchvision.transforms', None), 'functional_tensor'):
                # Create missing module that Real-ESRGAN expects
                from types import ModuleType
                functional_tensor = ModuleType('functional_tensor')
                # Copy functions from functional to functional_tensor
                for attr in dir(F):
                    if not attr.startswith('_'):
                        setattr(functional_tensor, attr, getattr(F, attr))
                sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor
            
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            print("📥 Loading Real-ESRGAN model...")
            
            # Use RealESRGAN_x2plus for 2x upscaling (512→1024)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            
            self.model = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=512,  # Process in tiles to avoid OOM
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == 'cuda' else False,
                device=self.device
            )
            
            self.model_loaded = True
            print("✅ Real-ESRGAN loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Could not load Real-ESRGAN: {e}")
            print("   Upscaling will use fallback bicubic interpolation")
            self.model_loaded = False
    
    def upscale_image(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """
        Upscale image using Real-ESRGAN or fallback to bicubic.
        
        Args:
            image: Input PIL Image (typically 512x512)
            scale: Upscaling factor (2 for 1024x1024, 4 for 2048x2048)
            
        Returns:
            Upscaled PIL Image
        """
        if not self.model_loaded:
            self.load_model()
        
        if self.model and self.model_loaded:
            try:
                # Convert PIL to numpy
                img_np = np.array(image)
                
                #  Real-ESRGAN inference
                output_np, _ = self.model.enhance(img_np, outscale=scale)
                
                # Convert back to PIL
                return Image.fromarray(output_np)
                
            except Exception as e:
                print(f"⚠️ Upscaling error: {e}, using fallback")
                # Fallback to bicubic
                new_size = (image.width * scale, image.height * scale)
                return image.resize(new_size, Image.BICUBIC)
        else:
            # Fallback: bicubic interpolation
            new_size = (image.width * scale, image.height * scale)
            return image.resize(new_size, Image.BICUBIC)


class QualityEnhancer:
    """
    Advanced image quality enhancements.
    Implements: Noise Reduction, Color Correction, Sharpening
    """
    
    @staticmethod
    def denoise(image: Image.Image, strength: int = 10) -> Image.Image:
        """
        Remove noise using bilateral filter.
        
        Args:
            image: Input PIL Image
            strength: Denoising strength (5-20, higher = more aggressive)
            
        Returns:
            Denoised PIL Image
        """
        import cv2
        
        # Convert to numpy
        img_np = np.array(image)
        
        # Apply bilateral filter (preserves edges while removing noise)
        denoised = cv2.bilateralFilter(img_np, d=9, sigmaColor=strength*5, sigmaSpace=strength*5)
        
        return Image.fromarray(denoised)
    
    @staticmethod
    def auto_color_correct(image: Image.Image) -> Image.Image:
        """
        Automatic color correction and balance.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Color-corrected PIL Image
        """
        from PIL import ImageOps, ImageEnhance
        
        # Auto-contrast
        image = ImageOps.autocontrast(image, cutoff=1)
        
        # Slight saturation boost
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    @staticmethod
    def sharpen(image: Image.Image, amount: float = 1.5) -> Image.Image:
        """
        Sharpen image for enhanced detail.
        
        Args:
            image: Input PIL Image
            amount: Sharpening strength (1.0-3.0)
            
        Returns:
            Sharpened PIL Image
        """
        from PIL import ImageFilter, ImageEnhance
        
        # Apply unsharp mask
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(amount*100), threshold=3))
        
        # Additional sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(amount)
        
        return image
    
    @staticmethod
    def enhance_all(image: Image.Image, denoise: bool = True, color_correct: bool = True, 
                   sharpen: bool = True) -> Image.Image:
        """
        Apply all quality enhancements in optimal order.
        
        Args:
            image: Input PIL Image
            denoise: Apply noise reduction
            color_correct: Apply auto color correction
            sharpen: Apply sharpening
            
        Returns:
            Enhanced PIL Image
        """
        if denoise:
            image = QualityEnhancer.denoise(image)
        
        if color_correct:
            image = QualityEnhancer.auto_color_correct(image)
        
        if sharpen:
            image = QualityEnhancer.sharpen(image)
        
        return image

# ==============================================================================
# EVALUATION METRICS (Rubric Item 4: 15 marks)
# ==============================================================================

class EvaluationMetrics:
    """
    Comprehensive evaluation framework with multiple metrics:
    - CLIP Score: Semantic alignment between image and text
    - FID: Image quality and distribution similarity
    - Inception Score: Image quality and diversity
    - Structural Similarity (SSIM): Pixel-level comparison
    """
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = config.device
        
        # Load CLIP Model
        self.clip_model = CLIPModel.from_pretrained(config.clip_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model)
        
        # Initialize FID and IS metrics
        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)
        self.inception_score = InceptionScore(normalize=True).to(self.device)
        
        # Results storage
        self.results_history: List[Dict] = []
        
    def calculate_clip_score(self, image: Image.Image, prompt: str) -> float:
        """
        Calculate CLIP score for image-text alignment.
        
        Args:
            image: Generated image
            prompt: Text prompt used for generation
            
        Returns:
            CLIP similarity score (higher is better)
        """
        try:
            inputs = self.clip_processor(
                text=[prompt], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                
            # Extract similarity score
            score = outputs.logits_per_image.item()
            return round(score, 3)
            
        except Exception as e:
            print(f"⚠️ CLIP Score Error: {e}")
            return 0.0
    
    def calculate_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two images.
        
        Args:
            img1, img2: Images to compare
            
        Returns:
            SSIM score (0-1, higher is better)
        """
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to grayscale numpy arrays
        img1_gray = np.array(img1.convert('L'))
        img2_gray = np.array(img2.convert('L'))
        
        score = ssim(img1_gray, img2_gray)
        return round(score, 3)
    
    def update_fid_metric(self, real_images: List[Image.Image], 
                         generated_images: List[Image.Image]):
        """
        Update FID calculation with new batch of images.
        
        Args:
            real_images: Original/reference images
            generated_images: Model-generated images
        """
        def images_to_tensor(images):
            """Convert list of PIL images to tensor"""
            tensors = []
            for img in images:
                img = img.resize((299, 299))  # Inception input size
                arr = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
                tensors.append(torch.from_numpy(arr))
            return torch.stack(tensors).to(self.device)
        
        real_tensor = images_to_tensor(real_images)
        gen_tensor = images_to_tensor(generated_images)
        
        self.fid.update(real_tensor, real=True)
        self.fid.update(gen_tensor, real=False)
    
    def compute_all_metrics(self, original_img: Image.Image, 
                          generated_img: Image.Image, 
                          prompt: str) -> Dict[str, float]:
        """
        Compute comprehensive metrics for a single generation.
        
        Args:
            original_img: Source image
            generated_img: Model output
            prompt: Generation prompt
            
        Returns:
            Dictionary of all metric scores
        """
        metrics = {
            'clip_score': self.calculate_clip_score(generated_img, prompt),
            'ssim': self.calculate_ssim(original_img, generated_img) if original_img else None,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results_history.append(metrics)
        return metrics
    
    def get_fid_score(self) -> float:
        """Compute and return final FID score"""
        try:
            fid_value = self.fid.compute().item()
            return round(fid_value, 3)
        except:
            return None
    
    def save_metrics_report(self, filepath: Path):
        """
        Save comprehensive metrics report to JSON.
        
        Args:
            filepath: Output JSON file path
        """
        report = {
            'total_generations': len(self.results_history),
            'average_clip_score': np.mean([r['clip_score'] for r in self.results_history]),
            'fid_score': self.get_fid_score(),
            'detailed_results': self.results_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"✅ Metrics report saved to {filepath}")

# ==============================================================================
# MODEL PIPELINES (Rubric Item 3: 15 marks)
# ==============================================================================

class AttributeControlPipeline:
    """
    Main pipeline orchestrator managing both control approaches.
    Implements factory pattern for model loading and dependency injection.
    """
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        # Model components
        self.pose_pipeline = None
        self.edit_pipeline = None
        self.pose_detector = None
        
        # Evaluation
        self.metrics = EvaluationMetrics(config)
        
        # Data management
        self.dataset_manager = DatasetManager(config)
        
        # Quality Enhancement (Phase 1)
        self.upscaler = SuperResolutionEnhancer(device=config.device)
        self.quality_enhancer = QualityEnhancer()
        
    def setup_models(self):
        """
        Initialize all model pipelines with error handling.
        Implements lazy loading for memory efficiency.
        """
        print("=" * 70)
        print("🚀 INITIALIZING ATTRIBUTE CONTROL SYSTEM")
        print("=" * 70)
        
        # Verify checkpoint
        if not os.path.exists(self.config.checkpoint_path):
            print(f"⚠️ WARNING: Local checkpoint not found at {self.config.checkpoint_path}")
            print("📥 Falling back to Stable Diffusion 1.5 from HuggingFace")
            self.config.checkpoint_path = "runwayml/stable-diffusion-v1-5"
        else:
            print(f"✅ Found local checkpoint: {self.config.checkpoint_path}")
        
        # Load Pose Control Pipeline
        self._setup_pose_pipeline()
        
        # Load Attribute Editing Pipeline
        self._setup_edit_pipeline()
        
        print("\n✅ ALL PIPELINES INITIALIZED SUCCESSFULLY\n")
        
    def _setup_pose_pipeline(self):
        """Load ControlNet-based pose control pipeline"""
        try:
            print("\n📦 Loading Pipeline A: ControlNet (Pose Control)")
            print("-" * 70)
            
            # Load ControlNet weights
            controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_model,
                torch_dtype=self.dtype
            )
            
            # Load main diffusion model
            if "safetensors" in self.config.checkpoint_path:
                self.pose_pipeline = StableDiffusionControlNetPipeline.from_single_file(
                    self.config.checkpoint_path,
                    controlnet=controlnet,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                ).to(self.device)
            else:
                self.pose_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    self.config.checkpoint_path,
                    controlnet=controlnet,
                    torch_dtype=self.dtype
                ).to(self.device)
            
            # Load pose detector
            self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            
            # Disable progress bars to prevent UI blocking
            self.pose_pipeline.set_progress_bar_config(disable=True)
            
            print("✅ ControlNet Pipeline Ready")
            print(f"   - Model: {self.config.controlnet_model}")
            print(f"   - Device: {self.device}")
            print(f"   - Precision: {self.dtype}")
            
        except Exception as e:
            print(f"❌ Error loading Pose Pipeline: {e}")
            raise
    
    def _setup_edit_pipeline(self):
        """Load InstructPix2Pix editing pipeline"""
        try:
            print("\n📦 Loading Pipeline B: InstructPix2Pix (Attribute Editing)")
            print("-" * 70)
            
            self.edit_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.config.instruct_pix2pix_model,
                torch_dtype=self.dtype
            ).to(self.device)
            
            # Disable progress bars to prevent UI blocking
            self.edit_pipeline.set_progress_bar_config(disable=True)
            
            print("✅ InstructPix2Pix Pipeline Ready")
            print(f"   - Model: {self.config.instruct_pix2pix_model}")
            print(f"   - Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading Edit Pipeline: {e}")
            raise
    
    def generate_with_pose_control(
        self, 
        input_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """
        Generate image with pose control using ControlNet.
        
        Args:
            input_image: Reference image for pose extraction
            prompt: Text description of desired output
            negative_prompt: Elements to avoid
            num_steps: Diffusion steps (quality vs speed tradeoff)
            guidance_scale: Adherence to prompt (higher = more literal)
            seed: Random seed for reproducibility
            
        Returns:
            (pose_skeleton, generated_image, metrics_dict)
        """
        if input_image is None:
            raise ValueError("Input image required for pose control")
        
        # Preprocess input
        input_image = self.dataset_manager.preprocess_image(input_image)
        
        # Extract pose skeleton
        print("🎭 Extracting pose skeleton...")
        pose_skeleton = self.pose_detector(input_image)
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate image
        print(f"🎨 Generating image (Steps: {num_steps}, CFG: {guidance_scale})...")
        
        # Use proper autocast context for PyTorch 2.x
        if self.device == "cuda":
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                output = self.pose_pipeline(
                    prompt=prompt,
                    image=pose_skeleton,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    callback_on_step_end=None  # Disable progress callbacks to prevent UI blocking
                ).images[0]
        else:
            output = self.pose_pipeline(
                prompt=prompt,
                image=pose_skeleton,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=None
            ).images[0]
        
        # Evaluate
        metrics = self.metrics.compute_all_metrics(input_image, output, prompt)
        
        # Clear GPU memory to prevent OOM errors
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return pose_skeleton, output, metrics
    
    def edit_with_instruction(
        self,
        input_image: Image.Image,
        instruction: str,
        num_steps: int = 20,
        text_guidance: float = 7.5,
        image_guidance: float = 1.5,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        """
        Edit image attributes using natural language instructions.
        
        Args:
            input_image: Image to edit
            instruction: Natural language editing command
            num_steps: Diffusion steps
            text_guidance: Strength of text conditioning
            image_guidance: Preservation of original structure
            seed: Random seed
            
        Returns:
            (edited_image, metrics_dict)
        """
        if input_image is None:
            raise ValueError("Input image required for editing")
        
        # Preprocess
        input_image = self.dataset_manager.preprocess_image(input_image)
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate edit
        print(f"✏️ Applying edit: '{instruction}'...")
        
        # Use proper autocast context for PyTorch 2.x
        if self.device == "cuda":
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                output = self.edit_pipeline(
                    prompt=instruction,
                    image=input_image,
                    num_inference_steps=num_steps,
                    guidance_scale=text_guidance,
                    image_guidance_scale=image_guidance,
                    callback_on_step_end=None  # Disable progress callbacks to prevent UI blocking
                ).images[0]
        else:
            output = self.edit_pipeline(
                prompt=instruction,
                image=input_image,
                num_inference_steps=num_steps,
                guidance_scale=text_guidance,
                image_guidance_scale=image_guidance,
                callback_on_step_end=None
            ).images[0]
        
        # Evaluate
        metrics = self.metrics.compute_all_metrics(input_image, output, instruction)
        
        # Clear GPU memory to prevent OOM errors
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return output, metrics

# ==============================================================================
# GRADIO USER INTERFACE (Rubric Item 6: 10 marks)
# ==============================================================================

def create_gradio_interface(pipeline: AttributeControlPipeline) -> gr.Blocks:
    """
    Create professional Gradio interface with comprehensive controls.
    
    Args:
        pipeline: Initialized AttributeControlPipeline
        
    Returns:
        Gradio Blocks interface
    """
    
    
    with gr.Blocks() as demo:
        gr.HTML("""
        <style>
        #col-container {
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """)
        
        gr.Markdown("""
        # 🎨 Latent Diffusion with Attribute Control
        ### Final Project - Generative AI (Spring 2025)
        **Comparative Analysis:** Architectural Control (ControlNet) vs. Instruction-Based Control (InstructPix2Pix)
        
        ---
        """)
        
        with gr.Tabs() as tabs:
            # ========== TAB 1: POSE CONTROL ==========
            with gr.TabItem("🎭 Pose Control (ControlNet)", id=0):
                gr.Markdown("""
                ### Architectural Control using ControlNet
                Upload a reference image to extract its pose skeleton, then generate a new image 
                matching that pose with your custom prompt.
                
                **Use Cases:** Character design, animation keyframes, pose-specific generation
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        pose_input = gr.Image(type="pil", label="📸 Reference Pose Image")
                        
                        # Pose Template Selector (Phase 3)
                        gr.Markdown("### 📐 Or Use Pose Template (Phase 3)")
                        with gr.Row():
                            pose_template_front = gr.Button("⬆️ Front Facing", size="sm")
                            pose_template_34 = gr.Button("↗️ 3/4 View", size="sm")
                            pose_template_profile = gr.Button("➡️ Profile", size="sm")
                        pose_template_status = gr.Textbox(label="Template Status", interactive=False, lines=1, visible=False)
                        
                        pose_prompt = gr.Textbox(
                            label="✍️ Generation Prompt",
                            value="A professional photograph of a doctor in a white coat, hospital background, high quality, 8k, realistic",
                            lines=3
                        )
                        pose_neg = gr.Textbox(
                            label="🚫 Negative Prompt",
                            value="deformed, distorted, disfigured, bad anatomy, extra limbs, blurry, low quality",
                            lines=2
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            pose_steps = gr.Slider(
                                minimum=10, maximum=50, value=30, step=1,
                                label="Inference Steps (Quality vs Speed)"
                            )
                            pose_guidance = gr.Slider(
                                minimum=1, maximum=20, value=7.5, step=0.5,
                                label="CFG Scale (Prompt Adherence)"
                            )
                            pose_seed = gr.Number(
                                label="Random Seed (Leave blank for random)",
                                value=None, precision=0
                            )
                        
                        pose_btn = gr.Button("🎨 Generate", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        pose_skeleton = gr.Image(label="🦴 Detected Pose Skeleton")
                        pose_output = gr.Image(label="✨ Generated Output")
                        pose_metrics = gr.JSON(label="📊 Evaluation Metrics", elem_classes="metric-box")
                
                # Template loading helper
                def load_pose_template(template_name):
                    """Load pose template JSON and convert to skeleton image"""
                    import json
                    from pathlib import Path
                    
                    template_path = Path("pose_templates") / f"{template_name}.json"
                    
                    if not template_path.exists():
                        return None, f"❌ Template not found: {template_path}"
                    
                    try:
                        # Load template
                        with open(template_path, 'r') as f:
                            template_data = json.load(f)
                        
                        # Create blank canvas
                        import cv2
                        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
                        
                        # Draw pose skeleton from keypoints
                        keypoints = template_data['keypoints']
                        
                        # OpenPose connections (simplified)
                        connections = [
                            (0, 1),   # nose-neck
                            (1, 2),   # neck-rshoulder
                            (2, 3),   # rshoulder-relbow
                            (3, 4),   # relbow-rwrist
                            (1, 5),   # neck-lshoulder
                            (5, 6),   # lshoulder-lelbow
                            (6, 7),   # lelbow-lwrist
                            (1, 8),   # neck-rhip
                            (8, 9),   # rhip-rknee
                            (9, 10),  # rknee-rankle
                            (1, 11),  # neck-lhip
                            (11, 12), # lhip-lknee
                            (12, 13), # lknee-lankle
                            (0, 14),  # nose-reye
                            (0, 15),  # nose-leye
                            (14, 16), # reye-rear
                            (15, 17)  # leye-lear
                        ]
                        
                        # Draw connections
                        for start_idx, end_idx in connections:
                            if start_idx < len(keypoints) and end_idx < len(keypoints):
                                start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                                end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                                confidence = min(keypoints[start_idx][2], keypoints[end_idx][2])
                                
                                if confidence > 0.3:  # Only draw if confident
                                    color_intensity = int(255 * confidence)
                                    cv2.line(canvas, start_pt, end_pt, (0, color_intensity, 0), 2)
                        
                        # Draw keypoints
                        for kp in keypoints:
                            if kp[2] > 0.3:  # confidence threshold
                                pt = (int(kp[0]), int(kp[1]))
                                color_intensity = int(255 * kp[2])
                                cv2.circle(canvas, pt, 4, (0, 0, color_intensity), -1)
                        
                        # Convert to PIL Image
                        skeleton_img = Image.fromarray(canvas)
                        
                        return skeleton_img, skeleton_img, f"✅ Loaded: {template_data['name']} - {template_data['description']}"
                        
                    except Exception as e:
                        return None, None, f"❌ Error loading template: {str(e)}"
                
                # Template button handlers
                pose_template_front.click(
                    lambda: load_pose_template("front_facing"),
                    outputs=[pose_input, pose_skeleton, pose_template_status]
                )
                
                pose_template_34.click(
                    lambda: load_pose_template("three_quarter"),
                    outputs=[pose_input, pose_skeleton, pose_template_status]
                )
                
                pose_template_profile.click(
                    lambda: load_pose_template("profile"),
                    outputs=[pose_input, pose_skeleton, pose_template_status]
                )
                
                # Main generation handler
                def pose_generation_wrapper(img, prompt, neg, steps, guid, seed):
                    if img is None:
                        return None, None, {"error": "Please upload a reference image"}
                    
                    try:
                        skeleton, output, metrics = pipeline.generate_with_pose_control(
                            input_image=img,
                            prompt=prompt,
                            negative_prompt=neg,
                            num_steps=int(steps),
                            guidance_scale=guid,
                            seed=int(seed) if seed else None
                        )
                        return skeleton, output, metrics
                    except Exception as e:
                        return None, None, {"error": str(e)}
                
                pose_btn.click(
                    fn=pose_generation_wrapper,
                    inputs=[pose_input, pose_prompt, pose_neg, pose_steps, pose_guidance, pose_seed],
                    outputs=[pose_skeleton, pose_output, pose_metrics],
                    queue=False  # Disable queue to prevent browser blocking
                )
            
            # ========== TAB 2: ATTRIBUTE EDITING ==========
            with gr.TabItem("✏️ Attribute Editing (InstructPix2Pix)", id=1):
                gr.Markdown("""
                ### Instruction-Based Control using InstructPix2Pix
                Upload an image and describe the changes you want using natural language.
                
                **Use Cases:** Lighting adjustment, color changes, style transfer, expression editing
                
                **Example Instructions:**
                - "Make the lighting warm and golden hour"
                - "Change hair color to blonde"
                - "Add dramatic blue cinematic lighting from the left"
                - "Make the person smile"
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        edit_input = gr.Image(type="pil", label="📸 Input Image")
                        edit_instruction = gr.Textbox(
                            label="💬 Edit Instruction",
                            placeholder="e.g., 'Make the lighting warm and golden' or 'Change hair color to red'",
                            lines=2
                        )
                        
                        with gr.Accordion("⚙️ Control Settings", open=True):
                            edit_steps = gr.Slider(
                                minimum=10, maximum=50, value=25,  # Increased from 20
                                step=1,
                                label="Inference Steps"
                            )
                            edit_text_cfg = gr.Slider(
                                minimum=1, maximum=20, value=9.5,  # Increased from 7.5 for stronger edits
                                step=0.5,
                                label="Text Guidance (Edit Strength)"
                            )
                            edit_img_cfg = gr.Slider(
                                minimum=1.0, maximum=5.0, value=1.3,  # Reduced from 1.5 for better preservation
                                step=0.1,
                                label="Image Guidance (Structure Preservation)"
                            )
                            edit_seed = gr.Number(
                                label="Random Seed",
                                value=None, precision=0
                            )
                        
                        # Preset buttons - Organized by category
                        gr.Markdown("### 🎨 Quick Presets")
                        
                        with gr.Accordion("🌟 Style Presets (Phase 2)", open=False):
                            gr.Markdown("**One-click professional styles:**")
                            with gr.Row():
                                preset_professional = gr.Button("💼 Professional", size="sm")
                                preset_casual = gr.Button("😊 Casual", size="sm")
                            with gr.Row():
                                preset_artistic = gr.Button("� Artistic", size="sm")
                                preset_fashion = gr.Button("👗 Fashion", size="sm")
                        
                        with gr.Accordion("💇 Attribute Presets (Phase 1)", open=False):
                            gr.Markdown("**Hair Colors:**")
                            with gr.Row():
                                preset_hair_blonde = gr.Button("Blonde", size="sm")
                                preset_hair_brown = gr.Button("Brown", size="sm")
                                preset_hair_black = gr.Button("Black", size="sm")
                                preset_hair_red = gr.Button("Red", size="sm")
                            
                            gr.Markdown("**Eye Colors:**")
                            with gr.Row():
                                preset_eye_blue = gr.Button("Blue Eyes", size="sm")
                                preset_eye_green = gr.Button("Green Eyes", size="sm")
                                preset_eye_brown = gr.Button("Brown Eyes", size="sm")
                                preset_eye_hazel = gr.Button("Hazel Eyes", size="sm")
                            
                            gr.Markdown("**Skin Tones:**")
                            with gr.Row():
                                preset_skin_fair = gr.Button("Fair", size="sm")
                                preset_skin_medium = gr.Button("Medium", size="sm")
                                preset_skin_dark = gr.Button("Dark", size="sm")
                            
                            gr.Markdown("**Lip Colors:**")
                            with gr.Row():
                                preset_lip_natural = gr.Button("Natural", size="sm")
                                preset_lip_red = gr.Button("Red", size="sm")
                                preset_lip_pink = gr.Button("Pink", size="sm")
                                preset_lip_dark = gr.Button("Dark", size="sm")
                        
                        with gr.Accordion("💡 Lighting Presets (Phase 1)", open=True):
                            gr.Markdown("**Lighting Style:**")
                            with gr.Row():
                                preset_warm = gr.Button("🌅 Warm", size="sm")
                                preset_cool = gr.Button("❄️ Cool", size="sm")
                                preset_dramatic = gr.Button("🎬 Dramatic", size="sm")
                            
                            gr.Markdown("**Light Direction:**")
                            with gr.Row():
                                preset_light_front = gr.Button("⬆️ Front Light", size="sm")
                                preset_light_side = gr.Button("➡️ Side Light", size="sm")
                                preset_light_back = gr.Button("⬇️ Back Light", size="sm")
                            
                            gr.Markdown("**Light Intensity:**")
                            with gr.Row():
                                preset_light_soft = gr.Button("🌙 Soft", size="sm")
                                preset_light_medium = gr.Button("☀️ Medium", size="sm")
                                preset_light_harsh = gr.Button("⚡ Harsh", size="sm")
                        
                        with gr.Accordion("😊 Expression Control (Phase 4)", open=True):
                            gr.Markdown("**Facial Expressions:**")
                            with gr.Row():
                                preset_expr_smile = gr.Button("😊 Gentle Smile", size="sm")
                                preset_expr_neutral = gr.Button("😐 Neutral", size="sm")
                                preset_expr_serious = gr.Button("😑 Serious", size="sm")
                            with gr.Row():
                                preset_expr_friendly = gr.Button("😃 Friendly", size="sm")
                                preset_expr_confident = gr.Button("😎 Confident", size="sm")
                        
                        edit_btn = gr.Button("✨ Apply Edit", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        edit_output = gr.Image(label="✨ Edited Result")
                        edit_metrics = gr.JSON(label="📊 Evaluation Metrics", elem_classes="metric-box")
                        
                        # Comparison slider
                        gr.Markdown("### 🔄 Before/After Comparison")
                        with gr.Row():
                            compare_before = gr.Image(label="Before", interactive=False)
                            compare_after = gr.Image(label="After", interactive=False)
                
                # Event handlers
                def edit_wrapper(img, instruction, steps, text_cfg, img_cfg, seed):
                    if img is None:
                        return None, {"error": "Please upload an image"}, None, None
                    if not instruction:
                        return None, {"error": "Please provide an edit instruction"}, None, None
                    
                    try:
                        output, metrics = pipeline.edit_with_instruction(
                            input_image=img,
                            instruction=instruction,
                            num_steps=int(steps),
                            text_guidance=text_cfg,
                            image_guidance=img_cfg,
                            seed=int(seed) if seed else None
                        )
                        return output, metrics, img, output
                    except Exception as e:
                        return None, {"error": str(e)}, None, None
                
                edit_btn.click(
                    fn=edit_wrapper,
                    inputs=[edit_input, edit_instruction, edit_steps, edit_text_cfg, edit_img_cfg, edit_seed],
                    outputs=[edit_output, edit_metrics, compare_before, compare_after],
                    queue=False  # Disable queue to prevent browser blocking
                )
                
                # ========== PRESET BUTTON HANDLERS ==========
                
                # Style Presets (Phase 2)
                preset_professional.click(
                    lambda: "Transform to professional corporate style with neutral lighting, front-facing composition, and polished appearance",
                    outputs=edit_instruction
                )
                preset_casual.click(
                    lambda: "Convert to casual relaxed style with natural warm lighting, friendly demeanor, and comfortable atmosphere",
                    outputs=edit_instruction
                )
                preset_artistic.click(
                    lambda: "Apply artistic creative style with dramatic lighting, unique angle, and expressive composition",
                    outputs=edit_instruction
                )
                preset_fashion.click(
                    lambda: "Transform to high-end fashion style with dramatic lighting, confident pose, and editorial aesthetic",
                    outputs=edit_instruction
                )
                
                # Hair Color Presets
                preset_hair_blonde.click(
                    lambda: "Change hair color to blonde, natural golden blonde tones, realistic hair texture",
                    outputs=edit_instruction
                )
                preset_hair_brown.click(
                    lambda: "Change hair color to brown, rich chocolate brown tones, natural appearance",
                    outputs=edit_instruction
                )
                preset_hair_black.click(
                    lambda: "Change hair color to black, deep jet black with subtle highlights, natural shine",
                    outputs=edit_instruction
                )
                preset_hair_red.click(
                    lambda: "Change hair color to red, vibrant auburn red tones, natural-looking color",
                    outputs=edit_instruction
                )
                
                # Eye Color Presets
                preset_eye_blue.click(
                    lambda: "Change eye color to blue, bright sapphire blue eyes, natural iris detail",
                    outputs=edit_instruction
                )
                preset_eye_green.click(
                    lambda: "Change eye color to green, emerald green eyes with natural depth",
                    outputs=edit_instruction
                )
                preset_eye_brown.click(
                    lambda: "Change eye color to brown, warm hazel brown eyes, natural appearance",
                    outputs=edit_instruction
                )
                preset_eye_hazel.click(
                    lambda: "Change eye color to hazel, multi-toned hazel eyes with green and brown mix",
                    outputs=edit_instruction
                )
                
                # Skin Tone Presets
                preset_skin_fair.click(
                    lambda: "Adjust skin tone to fair, light porcelain complexion with natural undertones",
                    outputs=edit_instruction
                )
                preset_skin_medium.click(
                    lambda: "Adjust skin tone to medium, warm medium complexion with even coloring",
                    outputs=edit_instruction
                )
                preset_skin_dark.click(
                    lambda: "Adjust skin tone to dark, rich deep complexion with natural glow",
                    outputs=edit_instruction
                )
                
                # Lip Color Presets
                preset_lip_natural.click(
                    lambda: "Apply natural lip color, subtle nude tones, natural finish",
                    outputs=edit_instruction
                )
                preset_lip_red.click(
                    lambda: "Apply red lip color, classic red lipstick, bold and vibrant",
                    outputs=edit_instruction
                )
                preset_lip_pink.click(
                    lambda: "Apply pink lip color, soft rose pink tones, fresh appearance",
                    outputs=edit_instruction
                )
                preset_lip_dark.click(
                    lambda: "Apply dark lip color, deep burgundy or plum tones, dramatic look",
                    outputs=edit_instruction
                )
                
                # Lighting Style Presets
                preset_warm.click(
                    lambda: "Apply warm lighting, golden hour glow, sunset tones, cozy atmosphere",
                    outputs=edit_instruction
                )
                preset_cool.click(
                    lambda: "Apply cool lighting, blue cinematic tones, moonlight effect, crisp atmosphere",
                    outputs=edit_instruction
                )
                preset_dramatic.click(
                    lambda: "Add cinematic dramatic lighting, moderate shadows, enhanced depth, professional film quality, preserve facial features",
                    outputs=edit_instruction
                )
                
                # Light Direction Presets
                preset_light_front.click(
                    lambda: "Apply front lighting, even illumination, soft shadows, professional headshot style",
                    outputs=edit_instruction
                )
                preset_light_side.click(
                    lambda: "Apply side lighting, dimensional shadows, sculptural quality, artistic portrait style",
                    outputs=edit_instruction
                )
                preset_light_back.click(
                    lambda: "Apply back lighting, rim light effect, glowing edges, dramatic silhouette",
                    outputs=edit_instruction
                )
                
                # Light Intensity Presets
                preset_light_soft.click(
                    lambda: "Apply soft lighting, diffused gentle light, minimal shadows, flattering glow",
                    outputs=edit_instruction
                )
                preset_light_medium.click(
                    lambda: "Apply medium lighting, balanced illumination, moderate shadows, natural appearance",
                    outputs=edit_instruction
                )
                preset_light_harsh.click(
                    lambda: "Apply strong direct lighting, defined shadows, dramatic effect, preserve details and features",
                    outputs=edit_instruction
                )
                
                # Expression Control Presets (Phase 4)
                preset_expr_smile.click(
                    lambda: "Add a gentle natural smile, subtle upturn of lips, warm friendly expression, preserve facial structure and identity",
                    outputs=edit_instruction
                )
                preset_expr_neutral.click(
                    lambda: "Change to neutral expression, relaxed face with no smile, calm demeanor, maintain facial features",
                    outputs=edit_instruction
                )
                preset_expr_serious.click(
                    lambda: "Change to serious professional expression, focused look, no smile, maintain composure and identity",
                    outputs=edit_instruction
                )
                preset_expr_friendly.click(
                    lambda: "Create friendly welcoming expression, warm genuine smile, approachable demeanor, preserve facial features",
                    outputs=edit_instruction
                )
                preset_expr_confident.click(
                    lambda: "Add confident expression, slight smile, strong eye contact, assertive demeanor, maintain facial structure",
                    outputs=edit_instruction
                )
            
            # ========== TAB 3: EVALUATION DASHBOARD ==========
            with gr.TabItem("📊 Evaluation Dashboard", id=2):
                gr.Markdown("""
                ### Model Performance Metrics
                Comprehensive evaluation of both control approaches using multiple metrics.
                """)
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
                    export_btn = gr.Button("💾 Export Report", variant="secondary")
                
                metrics_display = gr.JSON(label="Current Session Metrics")
                
                def get_current_metrics():
                    history = pipeline.metrics.results_history
                    if not history:
                        return {"message": "No generations yet. Try the Pose or Edit tabs first!"}
                    
                    return {
                        "total_generations": len(history),
                        "average_clip_score": round(np.mean([r['clip_score'] for r in history]), 3),
                        "recent_results": history[-5:]  # Last 5 generations
                    }
                
                def export_metrics_report():
                    filepath = pipeline.config.metrics_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    pipeline.metrics.save_metrics_report(filepath)
                    return {"status": "exported", "filepath": str(filepath)}
                
                refresh_btn.click(fn=get_current_metrics, outputs=metrics_display)
                export_btn.click(fn=export_metrics_report, outputs=metrics_display)
            
            # ========== TAB 4: BATCH PROCESSING & EXPORT ==========
            with gr.TabItem("🔄 Batch Processing & Export (Phase 2)", id=3):
                gr.Markdown("""
                ### Batch Generation and Enhanced Export
                Generate multiple variations and export with quality enhancements.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input = gr.Image(type="pil", label="📸 Input Image")
                        batch_instruction = gr.Textbox(
                            label="💬 Edit Instruction",
                            placeholder="e.g., 'Professional headshot style'",
                            lines=2
                        )
                        
                        batch_count = gr.Slider(
                            minimum=3, maximum=5, value=3, step=1,
                            label="Number of Variations"
                        )
                        
                        gr.Markdown("### ✨ Quality Enhancements")
                        enhance_sharpen = gr.Checkbox(label="🔪 Sharpen", value=False)
                        enhance_contrast = gr.Checkbox(label="🌓 Enhance Contrast", value=False)
                        
                        gr.Markdown("### 💾 Export Settings")
                        export_format = gr.Radio(
                            choices=["PNG", "JPEG"],
                            value="PNG",
                            label="Format"
                        )
                        export_quality = gr.Slider(
                            minimum=1, maximum=100, value=95, step=1,
                            label="JPEG Quality (if JPEG selected)"
                        )
                        
                        # Aspect Ratio Control (Phase 2)
                        export_aspect = gr.Radio(
                            choices=["Original", "Square (1:1)", "Portrait (3:4)", "Landscape (4:3)"],
                            value="Original",
                            label="Aspect Ratio"
                        )
                        
                        # ZIP Export Option
                        export_zip = gr.Checkbox(label="📦 Export as ZIP file", value=True)
                        
                        batch_btn = gr.Button("🚀 Generate Batch", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Variations")
                        batch_output1 = gr.Image(label="Variation 1")
                        batch_output2 = gr.Image(label="Variation 2")
                        batch_output3 = gr.Image(label="Variation 3")
                        batch_output4 = gr.Image(label="Variation 4", visible=False)
                        batch_output5 = gr.Image(label="Variation 5", visible=False)
                        
                        batch_status = gr.Textbox(label="Status", interactive=False)
                
                # Batch processing handler
                def batch_process(img, instruction, count, sharpen, contrast, fmt, quality, aspect, create_zip):
                    if img is None:
                        return [None]*5 + ["❌ Please upload an image"]
                    if not instruction:
                        return [None]*5 + ["❌ Please provide an instruction"]
                    
                    from PIL import ImageEnhance, ImageFilter
                    import random
                    import zipfile
                    
                    outputs = []
                    file_paths = []
                    
                    try:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        for i in range(int(count)):
                            # Generate with different seed
                            seed = random.randint(0, 999999)
                            output, metrics = pipeline.edit_with_instruction(
                                input_image=img,
                                instruction=instruction,
                                num_steps=20,
                                text_guidance=7.5,
                                image_guidance=1.5,
                                seed=seed
                            )
                            
                            # Apply aspect ratio transformation
                            if aspect != "Original":
                                if "Square" in aspect:
                                    # Center crop to square
                                    min_dim = min(output.width, output.height)
                                    left = (output.width - min_dim) // 2
                                    top = (output.height - min_dim) // 2
                                    output = output.crop((left, top, left + min_dim, top + min_dim))
                                elif "Portrait" in aspect:
                                    # 3:4 aspect ratio (portrait)
                                    new_height = int(output.width * 4 / 3)
                                    output = output.resize((output.width, new_height), Image.Resampling.LANCZOS)
                                elif "Landscape" in aspect:
                                    # 4:3 aspect ratio (landscape)
                                    new_width = int(output.height * 4 / 3)
                                    output = output.resize((new_width, output.height), Image.Resampling.LANCZOS)
                            
                            # Apply quality enhancements
                            if sharpen:
                                output = output.filter(ImageFilter.SHARPEN)
                            
                            if contrast:
                                enhancer = ImageEnhance.Contrast(output)
                                output = enhancer.enhance(1.2)
                            
                            # Save with export settings
                            filename = f"batch_{i+1}_{timestamp}.{fmt.lower()}"
                            filepath = pipeline.config.results_dir / filename
                            
                            if fmt == "JPEG":
                                output.save(filepath, format='JPEG', quality=int(quality))
                            else:
                                output.save(filepath, format='PNG')
                            
                            outputs.append(output)
                            file_paths.append(filepath)
                        
                        # Create ZIP if requested
                        if create_zip:
                            zip_path = pipeline.config.results_dir / f"batch_{timestamp}.zip"
                            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for fpath in file_paths:
                                    zipf.write(fpath, fpath.name)
                            status_msg = f"✅ Generated {int(count)} variations!\n📦 Exported to ZIP: {zip_path.name}\n📁 Individual files: {pipeline.config.results_dir}"
                        else:
                            status_msg = f"✅ Generated {int(count)} variations successfully!\n📁 Saved to: {pipeline.config.results_dir}"
                        
                        # Pad with None if less than 5
                        while len(outputs) < 5:
                            outputs.append(None)
                        
                        return outputs + [status_msg]
                        
                    except Exception as e:
                        return [None]*5 + [f"❌ Error: {str(e)}"]
                
                # Update visibility based on count
                def update_visibility(count):
                    return {
                        batch_output4: gr.update(visible=count >= 4),
                        batch_output5: gr.update(visible=count >= 5)
                    }
                
                batch_count.change(
                    fn=update_visibility,
                    inputs=[batch_count],
                    outputs=[batch_output4, batch_output5]
                )
                
                batch_btn.click(
                    fn=batch_process,
                    inputs=[batch_input, batch_instruction, batch_count, enhance_sharpen, 
                           enhance_contrast, export_format, export_quality, export_aspect, export_zip],
                    outputs=[batch_output1, batch_output2, batch_output3, 
                            batch_output4, batch_output5, batch_status],
                    queue=False
                )
            
            # ========== TAB 5: QUALITY ENHANCEMENT & UPSCALING (Phase 1) ==========
            with gr.TabItem("✨ Quality Enhancement & Upscaling", id=4):
                gr.Markdown("""
                ### Professional Quality Enhancement
                Upscale images to 1024x1024 or 2048x2048 with Real-ESRGAN and apply quality enhancements.
                
                **Features:**
                - 🔍 Super-Resolution: 512→1024 or 512→2048 using Real-ESRGAN
                - 🔇 Noise Reduction: Advanced bilateral filtering
                - 🎨 Color Correction: Automatic color balance and enhancement
                - 🔪 Sharpening: Enhanced detail preservation
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        quality_input = gr.Image(type="pil", label="📸 Input Image (any size)")
                        
                        gr.Markdown("### 🔍 Upscaling Options")
                        upscale_factor = gr.Radio(
                            choices=["None", "2x (512→1024)", "4x (512→2048)"],
                            value="2x (512→1024)",
                            label="Upscaling Factor"
                        )
                        
                        gr.Markdown("### ✨ Quality Enhancements")
                        quality_denoise = gr.Checkbox(label="🔇 Noise Reduction", value=True)
                        denoise_strength = gr.Slider(
                            minimum=5, maximum=20, value=10, step=1,
                            label="Noise Reduction Strength",
                            visible=True
                        )
                        
                        quality_color = gr.Checkbox(label="🎨 Auto Color Correct", value=True)
                        quality_sharpen = gr.Checkbox(label="🔪 Sharpen Details", value=True)
                        sharpen_amount = gr.Slider(
                            minimum=1.0, maximum=3.0, value=1.5, step=0.1,
                            label="Sharpening Amount",
                            visible=True
                        )
                        
                        gr.Markdown("### 💾 Save Options")
                        quality_format = gr.Radio(
                            choices=["PNG", "JPEG"],
                            value="PNG",
                            label="Output Format"
                        )
                        
                        quality_btn = gr.Button("🚀 Enhance & Upscale", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        quality_output = gr.Image(label="✨ Enhanced Output")
                        quality_info = gr.Textbox(label="Output Info", interactive=False, lines=3)
                        
                        gr.Markdown("### 🔄 Before/After Comparison")
                        with gr.Row():
                            quality_before = gr.Image(label="Before", interactive=False)
                            quality_after = gr.Image(label="After", interactive=False)
                
                # Quality enhancement handler
                def enhance_quality(img, upscale, denoise, denoise_str, color, sharpen, sharpen_amt, fmt):
                    if img is None:
                        return None, "❌ Please upload an image", None, None
                    
                    try:
                        import time
                        start_time = time.time()
                        
                        original_size = img.size
                        result = img.copy()
                        operations = []
                        
                        # Apply quality enhancements first (before upscaling for better results)
                        if denoise:
                            result = pipeline.quality_enhancer.denoise(result, strength=int(denoise_str))
                            operations.append(f"Noise Reduction (strength={denoise_str})")
                        
                        if color:
                            result = pipeline.quality_enhancer.auto_color_correct(result)
                            operations.append("Auto Color Correction")
                        
                        # Upscale
                        if upscale != "None":
                            scale = 2 if "2x" in upscale else 4
                            result = pipeline.upscaler.upscale_image(result, scale=scale)
                            operations.append(f"Upscaling {scale}x ({original_size[0]}x{original_size[1]} → {result.size[0]}x{result.size[1]})")
                        
                        # Sharpen after upscaling
                        if sharpen:
                            result = pipeline.quality_enhancer.sharpen(result, amount=sharpen_amt)
                            operations.append(f"Sharpening (amount={sharpen_amt})")
                        
                        # Save
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"enhanced_{timestamp}.{fmt.lower()}"
                        filepath = pipeline.config.results_dir / filename
                        
                        if fmt == "JPEG":
                            result.save(filepath, format='JPEG', quality=95)
                        else:
                            result.save(filepath, format='PNG')
                        
                        elapsed = time.time() - start_time
                        
                        info = f"✅ Enhancement Complete!\n"
                        info += f"⏱️ Processing Time: {elapsed:.2f}s\n"
                        info += f"📐 Output Size: {result.size[0]}x{result.size[1]}\n"
                        info += f"📁 Saved: {filename}\n\n"
                        info += "Operations Applied:\n" + "\n".join(f"• {op}" for op in operations)
                        
                        return result, info, img, result
                        
                    except Exception as e:
                        return None, f"❌ Error: {str(e)}", None, None
                
                # Toggle visibility based on checkboxes
                quality_denoise.change(
                    lambda x: gr.update(visible=x),
                    inputs=[quality_denoise],
                    outputs=[denoise_strength]
                )
                
                quality_sharpen.change(
                    lambda x: gr.update(visible=x),
                    inputs=[quality_sharpen],
                    outputs=[sharpen_amount]
                )
                
                quality_btn.click(
                    fn=enhance_quality,
                    inputs=[quality_input, upscale_factor, quality_denoise, denoise_strength,
                           quality_color, quality_sharpen, sharpen_amount, quality_format],
                    outputs=[quality_output, quality_info, quality_before, quality_after],
                    queue=False
                )
        
        gr.Markdown("""
        ---
        ### 📚 Project Information
        - **Course:** Generative AI (Spring 2025)
        - **Instructor:** Dr. Akhtar Jamil
        - **Students:** Muhammad Abdul Wahab Kiyani (22i-1178), Syed Ahmed Ali Zaidi (22i-1237)
        - **GitHub:** [Add your repository link here]
        """)
    
    return demo

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point for the application"""
    
    # Initialize configuration
    config = ProjectConfig()
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Latent Diffusion with Attribute Control                         ║
    ║  Final Project - Generative AI (Spring 2025)                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    System Configuration:
    - Device: {config.device}
    - Precision: {config.dtype}
    - Output Directory: {config.output_dir}
    """)
    
    # Initialize pipeline
    pipeline = AttributeControlPipeline(config)
    pipeline.setup_models()
    
    # Clear GPU memory after model loading
    if config.device == "cuda":
        print("\n🧹 Clearing GPU memory after model initialization...")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Print GPU memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"   GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # Create and launch interface
    print("\n🌐 Launching Gradio Interface...")
    demo = create_gradio_interface(pipeline)
    
    # Launch with public link for easy sharing
    demo.launch(
        share=True,  # Creates public URL
        server_name="0.0.0.0",  # Accessible from network
        server_port=7860,
        show_error=True,
        inbrowser=True  # Auto-open in browser
    )

if __name__ == "__main__":
    main()
