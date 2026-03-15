from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()

try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-preview-image-generation"
)

generation_config = {
    "response_modalities": ["TEXT", "IMAGE"]
}


class GenerateRequest(BaseModel):
    event_name: str = "TechXperts 2025"
    event_description: str = "A national-level hackathon bringing together innovators."


@app.post("/generate")
async def generate_image(payload: GenerateRequest):

    try:
        text_input = (
            f"A futuristic, collectible NFT memento token for the event: '{payload.event_name}'.\n\n"
            f"Core Concept: '{payload.event_description}'.\n\n"
            "Object & Form: A distinct 2D illustrated token, symbolic coin, crystal, holographic card, or futuristic emblem.\n"
            "Style: Futuristic, Web3, cyberpunk, neon gradients, vector-style, flat geometric.\n"
            "Background: Minimalist dark abstract background.\n"
            f"Text: Include '{payload.event_name}' subtly.\n"
            "Avoid: 3D renders, photorealism, blurry, cartoonish, watermark."
        )

        response = model.generate_content(
            contents=[text_input],
            generation_config=generation_config
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                image_data = part.inline_data.data
                image = Image.open(BytesIO(image_data))

                img_io = BytesIO()
                image.save(img_io, format="PNG")
                img_io.seek(0)

                return StreamingResponse(img_io, media_type="image/png")

        raise HTTPException(status_code=500, detail="No image generated")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "FastAPI Gemini Image Generator running"}