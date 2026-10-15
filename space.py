import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import spaces
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import search, content, alignment

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

if __name__ == "__main__":
    # 1. Launch Gradio server first (ZeroGPU hooks startup report and instantiates demo.server_app)
    demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

    # 2. Mount all backend API routes
    demo.server_app.include_router(search.router, prefix="/api/v1")
    demo.server_app.include_router(content.router, prefix="/api/v1")
    demo.server_app.include_router(alignment.router, prefix="/api/v1")

    @demo.server_app.get("/health")
    def health():
        return {"status": "ok", "mode": "gradio_zerogpu"}

    # 4. Keep server alive indefinitely
    demo.block_thread()
