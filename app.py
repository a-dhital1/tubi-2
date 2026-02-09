"""
Gradio app for Screenplay Generator - deploy to Hugging Face Spaces
"""

import torch
import gradio as gr
from screenplay_transformer import BPETokenizer, ScreenplayGPT, parse_prompt, format_output


# Load model and tokenizer at startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BPETokenizer.load('./tokenizer')
model = ScreenplayGPT.load('./checkpoints/best.pt', device=device).to(device)
model.eval()


def generate_screenplay(prompt, temperature, max_tokens, stop_at_scene):
    """Generate screenplay from natural language prompt."""
    if not prompt.strip():
        return "Please enter a scene description."
    
    # Parse and encode prompt
    formatted = parse_prompt(prompt)
    ids = tokenizer.encode(formatted)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    # Generate
    stop = tokenizer.scene_end_id if stop_at_scene else None
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            stop_at=stop
        )
    
    # Decode and format
    text = tokenizer.decode(out[0].tolist())
    return format_output(text)


# Gradio interface
demo = gr.Interface(
    fn=generate_screenplay,
    inputs=[
        gr.Textbox(
            label="Scene Description",
            placeholder="e.g., dark alley at night, detective searches for clues",
            lines=2
        ),
        gr.Slider(
            minimum=0.1, maximum=2.0, value=0.8, step=0.1,
            label="Temperature (higher = more creative)"
        ),
        gr.Slider(
            minimum=50, maximum=1000, value=300, step=50,
            label="Max Tokens"
        ),
        gr.Checkbox(
            label="Stop at scene end",
            value=True
        )
    ],
    outputs=gr.Textbox(label="Generated Screenplay", lines=20),
    title="🎬 Screenplay Generator",
    description="Generate movie screenplay scenes from natural language descriptions. Just describe a scene and the AI will write it in proper screenplay format.",
    examples=[
        ["dark alley at night, detective finds a clue", 0.8, 300, True],
        ["busy coffee shop in the morning", 0.7, 300, True],
        ["abandoned warehouse, two criminals argue", 0.9, 400, True],
    ],
    theme="soft"
)


if __name__ == "__main__":
    demo.launch()
