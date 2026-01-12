"""
Run inference for fine tuned Qwen3 model
"""

from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load model and tokenizer from Hugging Face Hub
# Alternatively, you can load from local files:
# model_name = "lora_model"  # For local loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="sachinkum0009/Qwen3-4B-LoRA"
)

# Enable inference mode for faster generation
FastLanguageModel.for_inference(model)

# Prepare the input message
messages = [{"role": "user", "content": "What is the derivative of ln(x^3 + 1)?"}]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # False
)

# Generate response
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)