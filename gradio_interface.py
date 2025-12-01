# ==============================================================================
# Gradio Interface Implementation
# ==============================================================================

import gradio as gr
from PIL import Image
from controlnet_pipeline import ControlNetPipeline, ProjectConfig
from instructpix2pix_pipeline import InstructPix2PixPipeline
from metrics_calculator import MetricsCalculator

class GradioInterface:
    """Interactive UI for latent diffusion attribute control"""
    
    def __init__(self):
        self.config = ProjectConfig()
        self.controlnet = ControlNetPipeline(self.config)
        self.instructpix2pix = InstructPix2PixPipeline(self.config)
        self.metrics = MetricsCalculator(self.config)
    
    def pose_control_generate(self, pose_image, prompt, steps, guidance):
        """Generate image with pose control"""
        if pose_image is None:
            return None, "Please upload a pose reference image"
        
        result = self.controlnet.generate(
            prompt=prompt,
            pose_image=pose_image,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        
        clip_score = self.metrics.calculate_clip_score(result, prompt)
        
        return result, f"CLIP Score: {clip_score:.2f}"
    
    def attribute_edit(self, image, instruction, steps, text_guidance, image_guidance):
        """Edit image with instruction"""
        if image is None:
            return None, None, "Please upload an image to edit"
        
        result = self.instructpix2pix.edit_image(
            image=image,
            instruction=instruction,
            num_inference_steps=steps,
            text_guidance_scale=text_guidance,
            image_guidance_scale=image_guidance
        )
        
        ssim_score = self.metrics.calculate_ssim(image, result)
        
        return image, result, f"SSIM: {ssim_score:.3f}"
    
    def create_interface(self):
        """Build Gradio interface"""
        
        with gr.Blocks(title="Latent Diffusion Attribute Control") as demo:
            gr.Markdown("# 🎨 Latent Diffusion with Attribute Control")
            gr.Markdown("**Compare ControlNet (Pose) vs InstructPix2Pix (Editing)**")
            
            with gr.Tabs():
                # Tab 1: Pose Control
                with gr.Tab("🎭 Pose Control (ControlNet)"):
                    with gr.Row():
                        with gr.Column():
                            pose_input = gr.Image(type="pil", label="Upload Pose Reference")
                            pose_prompt = gr.Textbox(
                                label="Generation Prompt",
                                placeholder="A professional doctor in a white coat"
                            )
                            pose_steps = gr.Slider(20, 50, value=30, step=1, label="Inference Steps")
                            pose_guidance = gr.Slider(5, 15, value=7.5, step=0.5, label="Guidance Scale")
                            pose_btn = gr.Button("🎨 Generate", variant="primary")
                        
                        with gr.Column():
                            pose_output = gr.Image(label="Generated Image")
                            pose_metrics = gr.Textbox(label="Metrics")
                    
                    pose_btn.click(
                        fn=self.pose_control_generate,
                        inputs=[pose_input, pose_prompt, pose_steps, pose_guidance],
                        outputs=[pose_output, pose_metrics]
                    )
                
                # Tab 2: Attribute Editing
                with gr.Tab("✏️ Attribute Editing (InstructPix2Pix)"):
                    with gr.Row():
                        with gr.Column():
                            edit_input = gr.Image(type="pil", label="Upload Image to Edit")
                            edit_instruction = gr.Textbox(
                                label="Edit Instruction",
                                placeholder="Change hair color to blonde"
                            )
                            
                            # Preset buttons
                            gr.Markdown("**Quick Presets:**")
                            with gr.Row():
                                btn_blonde = gr.Button("👱 Blonde Hair")
                                btn_warm = gr.Button("🌅 Warm Lighting")
                                btn_smile = gr.Button("😊 Gentle Smile")
                            
                            edit_steps = gr.Slider(10, 30, value=20, step=1, label="Inference Steps")
                            edit_text_guidance = gr.Slider(5, 15, value=7.5, step=0.5, label="Text Guidance")
                            edit_image_guidance = gr.Slider(1, 2, value=1.5, step=0.1, label="Image Guidance")
                            edit_btn = gr.Button("✨ Apply Edit", variant="primary")
                        
                        with gr.Column():
                            edit_before = gr.Image(label="Before")
                            edit_after = gr.Image(label="After")
                            edit_metrics = gr.Textbox(label="Metrics")
                    
                    # Preset button actions
                    btn_blonde.click(lambda: "Change hair color to blonde", outputs=edit_instruction)
                    btn_warm.click(lambda: "Make the lighting warm like sunset", outputs=edit_instruction)
                    btn_smile.click(lambda: "Change to a gentle smile", outputs=edit_instruction)
                    
                    edit_btn.click(
                        fn=self.attribute_edit,
                        inputs=[edit_input, edit_instruction, edit_steps, edit_text_guidance, edit_image_guidance],
                        outputs=[edit_before, edit_after, edit_metrics]
                    )
        
        return demo

# Launch interface
if __name__ == "__main__":
    interface = GradioInterface()
    demo = interface.create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
