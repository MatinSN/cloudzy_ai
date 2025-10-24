import os
import json
from openai import OpenAI


from dotenv import load_dotenv
load_dotenv()

class ImageDescriber:
    """
    Class for generating descriptive metadata (tags, description, caption)
    for an image using Hugging Face's inference endpoint via OpenAI client.
    """

    def __init__(self):
        # Read token from environment variable
        api_key = os.getenv("HF_TOKEN_1")
        if not api_key:
            raise ValueError("Environment variable HF_TOKEN_1 is not set.")

        # Initialize client
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )

        # Model to use
        self.model = "Qwen/Qwen3-VL-8B-Instruct:novita"

    def describe_image(self, image_url: str) -> dict:
        """
        Sends the image to the model and returns a structured dictionary:
        {
            "tags": [...],
            "description": "...",
            "caption": "..."
        }
        """
        # Prompt for structured output
        prompt = """
Describe this image in the following exact format:

result: {
  "tags": [list of tags related to the image],
  "description": "a 10-line descriptive description for the image",
  "caption": "a short description for the image"
}
"""
        

        # Send request
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )

        # Extract message text
        message = completion.choices[0].message
        text_content = message.content.strip()

        # Try to extract JSON-like dict from model output
        try:
            start = text_content.index("{")
            end = text_content.rindex("}") + 1
            json_str = text_content[start:end]
            result = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse model output: {text_content}\nError: {e}")

        return result


def main():
    """
    Entry point: takes image URL as input and prints parsed description.
    """
    describer = ImageDescriber()
    result = describer.describe_image("https://userx2000-cloudzy-ai-challenge.hf.space/uploads/img_2_20251024_082115_102.jpeg")
    print("\n✅ Extracted Result:\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
