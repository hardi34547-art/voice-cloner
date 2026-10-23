"""ElevenLabs Voice Cloning Web App - Backend"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from pathlib import Path
import tempfile
import httpx
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ELEVENLABS_API_KEY")
BASE_URL = "https://api.elevenlabs.io/v1"

app = FastAPI(title="Voice Cloner")

# Create output dir
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/api/add-voice")
async def add_voice(
    name: str = Form(...),
    audio: UploadFile = File(...)
):
    """Add new voice to ElevenLabs account"""
    try:
        if not API_KEY:
            return JSONResponse({"error": "API key not set"}, status_code=500)
        
        content = await audio.read()
        
        if not content:
            return JSONResponse({"error": "Empty file"}, status_code=400)
        
        print(f"[add-voice] name={name}, file={audio.filename}, size={len(content)}")
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/voices/add",
                headers={"xi-api-key": API_KEY},
                files={"files": (audio.filename, content, audio.content_type)},
                data={"name": name}
            )
        
        print(f"[add-voice] EL response: {response.status_code}")
        
        if response.status_code != 200:
            error_text = response.text
            print(f"[add-voice] ERROR: {error_text}")
            return JSONResponse(
                {"error": error_text},
                status_code=response.status_code
            )
        
        voice_data = response.json()
        return {"success": True, "voice_id": voice_data["voice_id"], "name": name}
    
    except Exception as e:
        print(f"[add-voice] EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/api/synthesize")
async def synthesize(
    text: str = Form(...),
    voice_id: str = Form(...)
):
    """Synthesize text with cloned voice"""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/text-to-speech/{voice_id}",
                headers={"xi-api-key": API_KEY},
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
            )
        
        if response.status_code != 200:
            return JSONResponse(
                {"error": response.text},
                status_code=response.status_code
            )
        
        # Save audio
        audio_path = OUTPUT_DIR / f"output_{Path(voice_id).name}_{hash(text)}.mp3"
        audio_path.write_bytes(response.content)
        
        return {
            "success": True,
            "audio_url": f"/api/download/{audio_path.name}"
        }
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/voices")
async def list_voices():
    """List all available voices"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/voices",
                headers={"xi-api-key": API_KEY}
            )
        
        if response.status_code != 200:
            return JSONResponse({"error": "Failed to fetch voices"}, status_code=400)
        
        voices = response.json()["voices"]
        return {
            "success": True,
            "voices": [
                {"id": v["voice_id"], "name": v["name"]}
                for v in voices
            ]
        }
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio"""
    audio_path = OUTPUT_DIR / filename
    if not audio_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(audio_path, media_type="audio/mpeg")

# Serve static frontend
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
