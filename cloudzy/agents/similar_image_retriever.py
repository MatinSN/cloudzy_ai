from smolagents import CodeAgent, OpenAIServerModel
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()


class ImageAnalyzerAgent:
    """Agent for analyzing images using Gemini with smolagents"""
    
    def __init__(self):
        """Initialize the agent with Gemini configuration"""
        # Configure Gemini with smolagents using OpenAI-compatible endpoint
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Get one at https://aistudio.google.com/apikey")
        
        # Use Gemini with smolagents via OpenAI-compatible API
        self.model = OpenAIServerModel(
            model_id="gemini-2.0-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        )
        
        # Instantiate the agent
        self.agent = CodeAgent(
            tools=[],
            model=self.model,
            max_steps=20,
            verbosity_level=2
        )
    
    def analyze_images(self, image_paths):
        """
        Load images from file paths and analyze them using the agent.
        
        Args:
            image_paths: List of Path objects or strings pointing to image files
            
        Returns:
            Agent response with image descriptions
        """
        # Convert strings to Path objects if needed
        image_paths = [Path(path) if isinstance(path, str) else path for path in image_paths]
        
        # Open and load images
        images = [Image.open(img_path) for img_path in image_paths if img_path.exists()]
        
        print(f"Loaded {len(images)} images from provided paths")
        
        if not images:
            print("No images found. Please provide valid image paths.")
            return None
        
        response = self.agent.run(
            """
            Describe these images to me:
            """,
            images=images
        )
        
        print("\n=== Agent Response ===")
        print(response)
        return response


# Test with sample images
if __name__ == "__main__":
    uploads_dir = Path(__file__).parent.parent.parent / "uploads"
    sample_image_paths = [
        uploads_dir / "img_1_20251024_180707_942.jpg",
        uploads_dir / "img_2_20251024_180749_372.jpeg",
        uploads_dir / "img_3_20251024_180756_356.jpeg",
    ]
    
    agent = ImageAnalyzerAgent()
    agent.analyze_images(sample_image_paths)