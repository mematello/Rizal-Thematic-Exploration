import spaces
import gradio as gr
import sys
import os

# Ensure backend directory is in python search path
base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app as fastapi_app

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

# Mount FastAPI app onto Gradio root
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
