# request_to_openai.py
import base64
import requests
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
API_KEY = os.getenv("OPENAI_KEY")
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def _encode_image(image_path):
    """
    Encode a local image file to base64 string (PNG format)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def gpt(
        text: str,
        model_name: str = "gpt-3.5-turbo",
        image_path: list = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 500
):
    """
    Call OpenAI Chat API (text or image) and return response safely.
    """
    if image_path is None:
        image_path = []

    # Prepare system and user content
    system_content = [{"type": "text", "text": system_prompt}]
    user_content = [{"type": "text", "text": text}]

    # Attach images if provided
    for img_path in image_path:
        try:
            b64_img = _encode_image(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_img}",
                    "detail": "low"
                }
            })
        except Exception as e:
            print(f"❌ Failed to encode image {img_path}: {e}")

    # Prepare payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        # Check if 'choices' exists
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            print(f"❌ No 'choices' in response: {response_json}")
            return "No response returned from OpenAI API."

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        return f"HTTP error: {e}"
    except Exception as e:
        print(f"❌ GPT request failed: {e}")
        return f"Error: {e}"
