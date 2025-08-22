#!/usr/bin/env python
"""Gradio interface for ZamAI Phi-3 Pashto model."""

import os
import sys

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from safety.filter import SafetyFilter
except ImportError:
    # Fallback if safety module not available
    class SafetyFilter:
        def check_text(self, text):
            return True, []


# Configuration
MODEL_ID = "tasal9/ZamZeerak-Phi3-Pashto"  # Update when model is published
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Global variables for model loading
model = None
tokenizer = None
safety_filter = SafetyFilter()


def load_model():
    """Load model and tokenizer."""
    global model, tokenizer
    if model is None:
        print(f"Loading model: {MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded successfully!")


def generate_response(
    prompt: str, max_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE, top_p: float = TOP_P
) -> str:
    """Generate response from the model."""
    load_model()

    # Safety check
    is_safe, violations = safety_filter.check_text(prompt)
    if not is_safe:
        return f"⚠️ Input flagged for safety concerns: {', '.join(violations)}"

    # Prepare prompt (add instruction format if needed)
    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Extract only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Safety check on output
        is_safe_output, output_violations = safety_filter.check_text(response)
        if not is_safe_output:
            return f"⚠️ Generated response flagged for safety: {', '.join(output_violations)}"

        return response

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


def create_interface():
    """Create Gradio interface."""

    with gr.Blocks(title="ZamAI Phi-3 Pashto", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🧠 ZamAI Phi-3 Mini Pashto</h1>
            <p>Pashto instruction-tuned language model based on Microsoft Phi-3 Mini</p>
            <p><em>د پښتو ژبې لپاره د هوښیارۍ ماډل</em></p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Your Message / ستاسو پیغام",
                    placeholder="Enter your question or instruction in Pashto or English...\nپښتو یا انګریزي کې خپله پوښتنه وليکئ...",
                    lines=4,
                    max_lines=10,
                )

                with gr.Row():
                    submit_btn = gr.Button("Generate / تولید", variant="primary")
                    clear_btn = gr.Button("Clear / پاک کړئ", variant="secondary")

                with gr.Accordion("Advanced Settings", open=False):
                    max_tokens = gr.Slider(minimum=50, maximum=1024, value=MAX_NEW_TOKENS, label="Max Tokens", step=50)
                    temperature = gr.Slider(minimum=0.1, maximum=2.0, value=TEMPERATURE, label="Temperature", step=0.1)
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=TOP_P, label="Top-p", step=0.05)

            with gr.Column(scale=3):
                output = gr.Textbox(
                    label="Generated Response / تولید شوی ځواب", lines=15, max_lines=20, interactive=False
                )

        # Examples
        with gr.Row():
            gr.Examples(
                examples=[
                    ["پښتو ژبه څه ده؟"],
                    ["د افغانستان د تاریخ په اړه راته ووایه"],
                    ["What is the capital of Afghanistan?"],
                    ["Tell me a story in Pashto"],
                    ["د ماشومانو لپاره یوه کیسه ولیکه"],
                ],
                inputs=prompt_input,
            )

        # Event handlers
        submit_btn.click(fn=generate_response, inputs=[prompt_input, max_tokens, temperature, top_p], outputs=output)

        clear_btn.click(fn=lambda: ("", ""), outputs=[prompt_input, output])

        prompt_input.submit(fn=generate_response, inputs=[prompt_input, max_tokens, temperature, top_p], outputs=output)

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #ddd;">
            <p>⚠️ This model is for research and educational purposes. Responses may not always be accurate.</p>
            <p>🔒 Basic safety filtering is applied to inputs and outputs.</p>
            <p>📝 Model: <a href="https://huggingface.co/tasal9/ZamZeerak-Phi3-Pashto" target="_blank">tasal9/ZamZeerak-Phi3-Pashto</a></p>
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
