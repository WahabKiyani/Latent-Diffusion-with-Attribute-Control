# ==============================================================================
# Super-Resolution and Quality Enhancement
# ==============================================================================

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch

class SuperResolutionEnhancer:
    """Implements upscaling using Real-ESRGAN"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
    
    def load_model(self):
        """Lazy loading of Real-ESRGAN model"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                           num_block=23, num_grow_ch=32, scale=4)
            
            self.model = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
            
            print("Real-ESRGAN model loaded successfully")
        except Exception as e:
            print(f"Failed to load Real-ESRGAN: {e}")
            print("Falling back to bicubic interpolation")
            self.model = None
    
    def upscale_image(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """
        Upscale image using Real-ESRGAN or fallback
        
        Args:
            image: Input PIL Image (typically 512x512)
            scale: Upscaling factor (2 or 4)
            
        Returns:
            Upscaled PIL Image
        """
        if self.model is None:
            self.load_model()
        
        # Convert to numpy
        img_array = np.array(image.convert('RGB'))
        
        if self.model is not None:
            try:
                # Use Real-ESRGAN
                output, _ = self.model.enhance(img_array, outscale=scale)
                return Image.fromarray(output)
            except Exception as e:
                print(f"Real-ESRGAN failed: {e}, using bicubic")
        
        # Fallback to bicubic
        new_size = (image.width * scale, image.height * scale)
        return image.resize(new_size, Image.BICUBIC)


class QualityEnhancer:
    """Advanced image quality enhancements"""
    
    @staticmethod
    def denoise(image: Image.Image, strength: int = 10) -> Image.Image:
        """Remove noise using bilateral filter"""
        img_array = np.array(image.convert('RGB'))
        denoised = cv2.bilateralFilter(img_array, 9, strength, strength)
        return Image.fromarray(denoised)
    
    @staticmethod
    def auto_color_correct(image: Image.Image) -> Image.Image:
        """Automatic color correction and balance"""
        img_array = np.array(image.convert('RGB')).astype(np.float32)
        
        # Normalize each channel
        for i in range(3):
            channel = img_array[:, :, i]
            channel = (channel - channel.min()) / (channel.max() - channel.min()) * 255
            img_array[:, :, i] = channel
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    @staticmethod
    def sharpen(image: Image.Image, factor: float = 1.5) -> Image.Image:
        """Apply sharpening filter"""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def enhance_all(image: Image.Image, denoise: bool = True, 
                   color_correct: bool = True, sharpen: bool = True) -> Image.Image:
        """Apply all enhancement filters"""
        result = image
        
        if denoise:
            result = QualityEnhancer.denoise(result)
        
        if color_correct:
            result = QualityEnhancer.auto_color_correct(result)
        
        if sharpen:
            result = QualityEnhancer.sharpen(result)
        
        return result


# Example usage
if __name__ == "__main__":
    print("Super-Resolution and Quality Enhancement modules loaded")
    print("Features:")
    print("- Real-ESRGAN upscaling (2x, 4x)")
    print("- Noise reduction")
    print("- Color correction")
    print("- Sharpening")
