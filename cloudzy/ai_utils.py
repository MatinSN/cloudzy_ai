import os
import numpy as np
from huggingface_hub import InferenceClient

from dotenv import load_dotenv
load_dotenv()

class ImageEmbeddingGenerator:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """
        Initialize the embedding generator with a Hugging Face model.
        """
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_TOKEN_1"],
        )
        self.model_name = model_name

    def generate_embedding(self, tags: list[str], description: str, caption: str) -> np.ndarray:
        """
        Generate a 512-d embedding for an image using its tags, description, and caption.

        Args:
            tags: List of tags related to the image
            description: Long descriptive text of the image
            caption: Short caption for the image

        Returns:
            embedding: 1D numpy array of shape (512,)
        """
        # Combine text fields into a single string
        text = " ".join(tags) + " " + description + " " + caption
        
        # Request embedding from Hugging Face
        result = self.client.feature_extraction(
            text,
            model=self.model_name,
        )
        
        # Convert to numpy array
        embedding = np.array(result, dtype=np.float32).reshape(-1)
        
        # Ensure shape is (512,)
        if embedding.shape[0] != 1024:
            raise ValueError(f"Expected embedding of size 512, got {embedding.shape[0]}")
        
        return embedding
    

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Internal helper to call Hugging Face feature_extraction and return a numpy array.
        """
        result = self.client.feature_extraction(
            text,
            model=self.model_name,
        )
        embedding = np.array(result, dtype=np.float32).reshape(-1)

        if embedding.shape[0] != 1024:
            raise ValueError(f"Expected embedding of size 1024, got {embedding.shape[0]}")
        return embedding


class TextSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the text summarizer with a Hugging Face model.
        """
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_TOKEN_1"],
        )
        self.model_name = model_name

    def summarize(self, text: str) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            
        Returns:
            summary: Generated summary string
        """
        if not text or text.strip() == "":
            return "Album of photos"
        
        try:
            result = self.client.summarization(
                text,
                model=self.model_name,
            )
            # Extract the summary text from the result object
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("summary_text", str(result[0]))
            elif isinstance(result, dict):
                return result.get("summary_text", str(result))
            else:
                return str(result)
        except Exception as e:
            # Fallback if summarization fails
            return f"Collection: {text[:80]}..."

# Example usage:
if __name__ == "__main__":
    generator = ImageEmbeddingGenerator()
    
    tags = ["nature", "sun", "ice cream"]
    description = "A sunny day in the park with children enjoying ice cream."
    caption = "Sunny day with ice cream."
    
    embedding = generator.generate_embedding(tags, description, caption)
    print("Embedding shape:", embedding.shape)
