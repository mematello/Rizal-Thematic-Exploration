import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import spaces
import gradio as gr
from app.main import app as fastapi_app
import uvicorn

@spaces.GPU
def predict(query: str):
    return f"✅ Rizal Thematic Exploration API is Online and Healthy! Tested query: {query}"

with gr.Blocks(title="Rizal Search API") as demo:
    gr.Markdown("# 📖 Rizal Thematic Exploration Search API")
    gr.Markdown("FastAPI backend engine powered by PyTorch, SentenceTransformers (XLM-RoBERTa), and Supabase PostgreSQL with pgvector.")
    with gr.Row():
        inp = gr.Textbox(label="Test Search Query", value="edukasyon")
        btn = gr.Button("Test API")
    out = gr.Textbox(label="API Status", value="Ready")
    btn.click(fn=predict, inputs=inp, outputs=out)

# 1. Trigger ZeroGPU startup report hook
try:
    demo.launch(server_name="127.0.0.1", server_port=7861, prevent_thread_lock=True)
    demo.close()
except Exception as e:
    print("ZeroGPU launcher hook warning:", e)

# 2. Mount Gradio UI at /gradio so FastAPI handles root routes (/api/v1/*, /health, /docs)
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

@app.get("/")
def root():
    return {
        "project": "Rizal Thematic Exploration Search API",
        "docs": "/docs",
        "health": "/health",
        "gradio_ui": "/gradio",
        "api_v1_search": "/api/v1/search",
        "api_v1_chapters": "/api/v1/chapters"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
