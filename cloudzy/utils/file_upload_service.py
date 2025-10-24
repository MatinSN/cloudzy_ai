import base64
import requests
from pathlib import Path
from typing import Optional
import os 


from dotenv import load_dotenv
load_dotenv()


class ImgBBUploader:
    """
    Upload an image file to ImgBB and return only the direct public URL.
    """

    API_ENDPOINT = "https://api.imgbb.com/1/upload"

    def __init__(self, expiration: Optional[int] = None, timeout: int = 30):
        api_key = os.getenv("IMAGE_UPLOAD_API_KEY")
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.expiration = expiration
        self.timeout = timeout

    def _encode_file_to_base64(self, image_path: Path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with image_path.open("rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("ascii")

    def upload(self, image_path: str) -> str:
        """
        Upload the image and return the direct URL of the uploaded file.
        """
        image_path_obj = Path(image_path)
        b64_image = self._encode_file_to_base64(image_path_obj)

        params = {"key": self.api_key}
        if self.expiration is not None:
            params["expiration"] = str(self.expiration)

        try:
            resp = requests.post(
                self.API_ENDPOINT,
                params=params,
                data={"image": b64_image},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            print(data)
            if data.get("success"):
                return data["data"]["url"]
            raise RuntimeError(f"Upload failed: {data}")
        except requests.RequestException as e:
            raise RuntimeError(f"ImgBB upload failed: {e}") from e


if __name__ == "__main__":
    uploader = ImgBBUploader(expiration=86400)
    result = uploader.upload("/Users/komeilfathi/Documents/hf_deploy_test/cloudzy_ai_challenge/uploads/img_1_20251024_080401_794.jpg")
    print(result)


        # resp = uploader.upload_file("/Users/komeilfathi/Documents/hf_deploy_test/cloudzy_ai_challenge/uploads/img_1_20251024_080401_794.jpg")





# Example usage:
# uploader = FreeImageHostUploader(api_key="your_api_key_here")
# image_url = uploader.upload_image("path_to_your_image.jpg")
# print(f"Image uploaded successfully: {image_url}")
