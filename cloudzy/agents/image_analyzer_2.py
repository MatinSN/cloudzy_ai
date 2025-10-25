from smolagents import CodeAgent, OpenAIServerModel
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()


class ImageAnalyzerAgent:
    """Agent for describing images using Gemini with smolagents"""
    
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
            max_steps=5,
            verbosity_level=1
        )
    
    def retrieve_similar_images(self, image_path):
        """
        Describe a given image.
        
        Args:
            image_path: Path object or string pointing to an image file
            
        Returns:
            Description text of the image
        """
        image_path = Path(image_path) if isinstance(image_path, str) else image_path
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        image = Image.open(image_path)
        print(f"Loaded image: {image_path.name}\n")
        
        response = self.agent.run(
            """
            Describe this image in a way that could be used as a prompt for generating a new image inspired by it.
Focus on the main subjects, composition, style, mood, and colors.
Avoid mentioning specific names or exact details — instead, describe the overall aesthetic and atmosphere so the result feels similar but not identical.
            """,
            images=[image]
        )
        
        return response
    
    def analyze_image_metadata(self, image_path):
        """
        Analyze an image and extract structured metadata (tags, description, caption).
        
        Args:
            image_path: Path object or string pointing to an image file
            
        Returns:
            Dictionary with keys: tags (list), description (str), caption (str)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If response cannot be parsed into valid JSON
        """
        image_path = Path(image_path) if isinstance(image_path, str) else image_path
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        image = Image.open(image_path)
        print(f"Loaded image: {image_path.name}\n")
        
        prompt = """
Describe this image in the following exact format:

result: {
  "tags": [list of tags related to the image],
  "description": "a 5-line descriptive description for the image",
  "caption": "a short description for the image"
}
        """
        
        response = self.agent.run(prompt, images=[image])
        
        # Ensure response is a string
        response_text = str(response) if response is not None else ""
        
        # Extract JSON part from response
        # Look for the pattern: result: { ... } (or { ... if closing brace is missing)
        match = re.search(r'result:\s*(\{[\s\S]*)', response_text)
        
        if not match:
            raise ValueError(f"Could not find JSON in response: {response_text}")
        
        json_str = match.group(1)
        
        # If the extracted JSON doesn't end with }, try adding it
        if not json_str.rstrip().endswith("}"):
            print(f"[Warning] No closing brace found in JSON, attempting to add closing brace...")
            json_str = json_str + "}"
        
        try:
            # Parse the JSON string into a dictionary
            result_dict = json.loads(json_str)
            return result_dict
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {json_str}\nError: {str(e)}")


# Test with sample images
if __name__ == "__main__":
    uploads_dir = Path(__file__).parent.parent.parent / "uploads"
    sample_image_paths = [
        uploads_dir / "img_1_20251024_180707_942.jpg",
        uploads_dir / "img_2_20251024_180749_372.jpeg",
        uploads_dir / "img_3_20251024_180756_356.jpeg",
    ]
    
    agent = ImageAnalyzerAgent()
    
    # Test with first sample image
    result = agent.retrieve_similar_images(sample_image_paths[0])
    print(f"\n=== Results ===")
    print(f"Description: {result}")
    # print(f"Similar images found: {len(result['similar_images'])}")