"""
ElevenLabs Voice Cloning - HuggingFace Spaces (Gradio)
Deploy to HuggingFace Spaces for free public access
"""
import gradio as gr
import httpx
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ELEVENLABS_API_KEY")
BASE_URL = "https://api.elevenlabs.io/v1"

# Create output dir
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_voices():
    """Fetch available voices from ElevenLabs"""
    try:
        response = httpx.get(
            f"{BASE_URL}/voices",
            headers={"xi-api-key": API_KEY},
            timeout=10
        )
        if response.status_code == 200:
            voices = response.json()["voices"]
            return {v["voice_id"]: v["name"] for v in voices}
        return {}
    except Exception as e:
        return {"error": str(e)}

def synthesize_voice(text, voice_id):
    """Synthesize text using ElevenLabs"""
    if not API_KEY:
        return None, "❌ API key not set. Add ELEVENLABS_API_KEY to secrets."
    
    if not text:
        return None, "❌ Text required"
    
    if not voice_id:
        return None, "❌ Select a voice"
    
    try:
        response = httpx.post(
            f"{BASE_URL}/text-to-speech/{voice_id}",
            headers={"xi-api-key": API_KEY},
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            },
            timeout=30
        )
        
        if response.status_code != 200:
            error = response.json().get("detail", {}).get("message", response.text)
            return None, f"❌ Error: {error}"
        
        # Save audio
        audio_path = OUTPUT_DIR / f"output_{hash(text)}.mp3"
        audio_path.write_bytes(response.content)
        
        return str(audio_path), "✅ Audio generated!"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def clone_voice(voice_name, audio_file):
    """Clone custom voice (requires paid plan)"""
    if not API_KEY:
        return "❌ API key not set"
    
    if not voice_name:
        return "❌ Voice name required"
    
    if not audio_file:
        return "❌ Audio file required"
    
    try:
        # Read audio file
        with open(audio_file, "rb") as f:
            audio_content = f.read()
        
        response = httpx.post(
            f"{BASE_URL}/voices/add",
            headers={"xi-api-key": API_KEY},
            files={"files": (Path(audio_file).name, audio_content)},
            data={"name": voice_name},
            timeout=60
        )
        
        if response.status_code != 200:
            error = response.json().get("detail", {}).get("message", response.text)
            return f"❌ {error}\n\n💡 Tip: Voice cloning requires ElevenLabs Paid Plan ($99+)"
        
        voice_id = response.json()["voice_id"]
        return f"✅ Voice '{voice_name}' cloned!\nVoice ID: {voice_id}"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load voices
voices_dict = get_voices()
voice_choices = list(voices_dict.values()) if voices_dict and "error" not in voices_dict else []

# Build Gradio Interface
with gr.Blocks(title="🎤 Voice Cloner - ElevenLabs", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎤 Voice Cloner with ElevenLabs")
    gr.Markdown("Synthesize speech & clone custom voices using AI. **Free tier available!**")
    
    with gr.Tabs():
        # Tab 1: Synthesize
        with gr.TabItem("✨ Synthesize"):
            gr.Markdown("Generate speech from text using preset voices")
            
            with gr.Row():
                with gr.Column():
                    voice_dropdown = gr.Dropdown(
                        choices=voice_choices,
                        label="Select Voice",
                        value=voice_choices[0] if voice_choices else None
                    )
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter text here...",
                        lines=4,
                        value="Halo, ini adalah suara kloning saya dari ElevenLabs."
                    )
                    synthesize_btn = gr.Button("🎵 Generate Audio", variant="primary")
                
                with gr.Column():
                    audio_output = gr.Audio(label="Generated Audio", type="filepath")
                    status_output = gr.Textbox(label="Status", interactive=False)
            
            synthesize_btn.click(
                fn=synthesize_voice,
                inputs=[text_input, voice_dropdown],
                outputs=[audio_output, status_output]
            )
        
        # Tab 2: Clone Voice
        with gr.TabItem("🎯 Clone Voice"):
            gr.Markdown("""
            Clone a custom voice from an audio sample.
            
            ⚠️ **Requires:** ElevenLabs Paid Plan ($99+/month)
            
            Free tier cannot clone custom voices.
            """)
            
            with gr.Row():
                with gr.Column():
                    voice_name_input = gr.Textbox(
                        label="Custom Voice Name",
                        placeholder="e.g., My Voice"
                    )
                    audio_upload = gr.Audio(
                        label="Voice Sample (10 sec - 5 min)",
                        type="filepath"
                    )
                    clone_btn = gr.Button("🔊 Clone Voice", variant="primary")
                
                with gr.Column():
                    clone_status = gr.Textbox(label="Status", interactive=False, lines=4)
            
            clone_btn.click(
                fn=clone_voice,
                inputs=[voice_name_input, audio_upload],
                outputs=clone_status
            )
        
        # Tab 3: Info
        with gr.TabItem("ℹ️ Info"):
            gr.Markdown(f"""
            ## Available Voices: {len(voice_choices)}
            
            Preset voices include:
            - **Jessica** - Female, playful, warm
            - **Daniel** - Male, broadcaster
            - **Brian** - Male, deep, resonant
            - **Roger** - Male, casual
            - ...and 17 more!
            
            ## Pricing
            | Plan | Cost | Limit | Cloning |
            |---|---|---|---|
            | **Free** | $0 | 10k chars/mo | ❌ |
            | **Starter** | $99/mo | 100k chars | ✅ |
            | **Pro** | $199/mo | 500k chars | ✅ |
            
            ## How to Use
            1. **Synthesize**: Pick a voice, type text, click "Generate Audio"
            2. **Clone**: Upload your voice sample (paid plan) to create custom voice
            3. **Download**: Use generated audio freely
            
            ## Links
            - [ElevenLabs](https://elevenlabs.io)
            - [API Docs](https://elevenlabs.io/docs)
            - [GitHub Repo](https://github.com/elevenlabs/api-spec)
            """)

demo.launch(share=True)
