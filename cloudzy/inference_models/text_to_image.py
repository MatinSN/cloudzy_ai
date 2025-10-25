import os
from datetime import datetime
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()


class TextToImageGenerator:
    """Class for generating images from text prompts using HuggingFace models"""
    
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-dev", provider: str = "nebius"):
        """
        Initialize the text-to-image generator.
        
        Args:
            model_id: HuggingFace model ID (default: FLUX.1-dev for high quality)
            provider: API provider (default: nebius)
        """
        api_key = os.getenv("HF_TOKEN_1")
        if not api_key:
            raise ValueError("HF_TOKEN_1 not found in environment variables")
        
        self.client = InferenceClient(
            provider=provider,
            api_key=api_key,
        )
        self.model_id = model_id
        self.uploads_dir = Path(__file__).parent.parent.parent / "uploads"
        self.uploads_dir.mkdir(exist_ok=True)
        
        self.app_domain = os.getenv("APP_DOMAIN", "http://127.0.0.1:8000/")
    
    def generate(self, prompt: str) -> str:
        """
        Generate an image from a text prompt and save it to the uploads folder.
        
        Args:
            prompt: Text description of the image to generate
            
        Returns:
            URL of the generated image in format: {APP_DOMAIN}uploads/{filename}
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Generate image using HuggingFace inference
            image = self.client.text_to_image(
                prompt,
                model=self.model_id,
            )
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"generated_{timestamp}.png"
            filepath = self.uploads_dir / filename
            
            # Save image
            image.save(filepath)
            
            # Return URL in the required format
            image_url = f"{self.app_domain}uploads/{filename}"
            return image_url
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {str(e)}") from e


# Test with sample prompt
if __name__ == "__main__":
    generator = TextToImageGenerator()
    
    # Test with a sample prompt
    prompt = "A beautiful sunset over mountains with birds flying"
    url = generator.generate(prompt)
    print(f"Generated image URL: {url}")