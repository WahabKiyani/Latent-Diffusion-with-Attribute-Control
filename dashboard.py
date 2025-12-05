# ==============================================================================
# Evaluation Dashboard Implementation
# ==============================================================================

import gradio as gr
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

class EvaluationDashboard:
    """Manages metrics tracking and visualization"""
    
    def __init__(self, metrics_dir: Path):
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
    
    def add_generation(self, method: str, prompt: str, clip_score: float, 
                      ssim_score: float = None, timestamp: str = None):
        """Add a generation to history"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        entry = {
            "method": method,
            "prompt": prompt,
            "clip_score": clip_score,
            "ssim_score": ssim_score,
            "timestamp": timestamp
        }
        
        self.history.append(entry)
    
    def get_statistics(self) -> Dict:
        """Calculate statistics from history"""
        if not self.history:
            return {
                "total_generations": 0,
                "avg_clip_score": 0.0,
                "controlnet_count": 0,
                "instructpix2pix_count": 0
            }
        
        controlnet_gens = [h for h in self.history if h["method"] == "ControlNet"]
        instruct_gens = [h for h in self.history if h["method"] == "InstructPix2Pix"]
        
        all_clip_scores = [h["clip_score"] for h in self.history if h["clip_score"] is not None]
        avg_clip = sum(all_clip_scores) / len(all_clip_scores) if all_clip_scores else 0.0
        
        return {
            "total_generations": len(self.history),
            "avg_clip_score": avg_clip,
            "controlnet_count": len(controlnet_gens),
            "instructpix2pix_count": len(instruct_gens)
        }
    
    def get_recent_history(self, n: int = 10) -> str:
        """Get recent generation history as formatted string"""
        if not self.history:
            return "No generations yet"
        
        recent = self.history[-n:]
        
        output = []
        for i, entry in enumerate(reversed(recent), 1):
            output.append(f"{i}. [{entry['method']}] {entry['prompt'][:50]}...")
            output.append(f"   CLIP: {entry['clip_score']:.2f}")
            if entry['ssim_score'] is not None:
                output.append(f"   SSIM: {entry['ssim_score']:.3f}")
            output.append("")
        
        return "\n".join(output)
    
    def export_report(self) -> str:
        """Export detailed metrics report to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_report_{timestamp}.json"
        filepath = self.metrics_dir / filename
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "history": self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return f"Report exported to {filepath}"
    
    def create_dashboard_tab(self) -> gr.Tab:
        """Create Gradio tab for evaluation dashboard"""
        
        with gr.Tab("📊 Evaluation Dashboard") as tab:
            gr.Markdown("## Generation Statistics and Metrics")
            
            with gr.Row():
                with gr.Column():
                    stats_display = gr.Textbox(
                        label="Overall Statistics",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Column():
                    history_display = gr.Textbox(
                        label="Recent Generation History",
                        lines=8,
                        interactive=False
                    )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
                export_btn = gr.Button("💾 Export Report", variant="primary")
            
            export_status = gr.Textbox(label="Export Status", interactive=False)
            
            def refresh_dashboard():
                stats = self.get_statistics()
                stats_text = f"""
Total Generations: {stats['total_generations']}
Average CLIP Score: {stats['avg_clip_score']:.2f}

Method Breakdown:
- ControlNet: {stats['controlnet_count']}
- InstructPix2Pix: {stats['instructpix2pix_count']}
                """.strip()
                
                history_text = self.get_recent_history(10)
                
                return stats_text, history_text
            
            def export_report_handler():
                message = self.export_report()
                return message
            
            refresh_btn.click(
                fn=refresh_dashboard,
                outputs=[stats_display, history_display]
            )
            
            export_btn.click(
                fn=export_report_handler,
                outputs=export_status
            )
        
        return tab


# Example usage
if __name__ == "__main__":
    dashboard = EvaluationDashboard(Path("outputs/metrics"))
    
    # Simulate some generations
    dashboard.add_generation("ControlNet", "A professional doctor", 28.5)
    dashboard.add_generation("InstructPix2Pix", "Change hair to blonde", 26.3, 0.85)
    
    print("Dashboard initialized with sample data")
    print(dashboard.get_statistics())
