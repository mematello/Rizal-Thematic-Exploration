import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, "backend")
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

# Mount FastAPI endpoints directly onto Gradio's internal FastAPI app
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
demo.app.include_router(search.router, prefix="/api/v1")
demo.app.include_router(content.router, prefix="/api/v1")
demo.app.include_router(alignment.router, prefix="/api/v1")

@demo.app.get("/health")
def health():
    return {"status": "ok", "mode": "gradio_zerogpu"}

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
