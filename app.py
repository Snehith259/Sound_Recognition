"""
SoundSense — Hugging Face Spaces & Production Cloud Entrypoint
Compatible with Hugging Face Spaces (Gradio SDK free tier), direct FastAPI, or Docker.
"""
import os
import uvicorn
from main import app as application

# If running on Hugging Face Spaces with Gradio SDK, mount Gradio bridge
try:
    import gradio as gr

    with gr.Blocks(title="SoundSense ESC-50 Acoustic Studio", theme=gr.themes.Soft()) as gradio_bridge:
        gr.Markdown("# 🔊 SoundSense — ESC-50 Acoustic Studio")
        gr.Markdown(
            "Welcome to SoundSense! The full interactive frontend is running at the root URL: "
            "[**Open SoundSense Studio**](/) \n\n"
            "*(This Gradio interface serves as the Hugging Face Spaces bridge)*"
        )
    app = gr.mount_gradio_app(application, gradio_bridge, path="/gradio")
except ImportError:
    # Standalone FastAPI deployment
    app = application

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
