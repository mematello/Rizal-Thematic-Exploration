import sys
import os

# Ensure backend directory is in python search path
base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import gradio as gr
from app.main import app as fastapi_app

def check_status():
    return "✅ Rizal Thematic Exploration API is Online and Healthy!\nAPI endpoints available at /api/v1/search and /api/v1/chapters."

with gr.Blocks(title="Rizal Search API") as demo:
    gr.Markdown("# 🇵🇭 Rizal Thematic Exploration Search API")
    gr.Markdown("FastAPI backend engine powered by PyTorch, SentenceTransformers (XLM-RoBERTa), and Supabase PostgreSQL with pgvector.")
    status_btn = gr.Button("Check Health Status")
    output = gr.Textbox(label="API Status", value="Ready")
    status_btn.click(fn=check_status, outputs=output)

# Mount FastAPI app onto Gradio root
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
