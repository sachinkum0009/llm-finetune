"""
Fine tune Qwen3 using Unsloth
"""

import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

max_seq_length = 2048
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")


def format_reasoning_data(examples):
    conversations = []
    for problem, solution in zip(examples["problem"], examples["generated_solution"]):
        convo = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
        # Apply template here to each individual conversation
        texts = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        conversations.append(texts)
    return {"text": conversations}


reasoning_dataset = reasoning_dataset.map(format_reasoning_data, batched=True)
reasoning_conversations = reasoning_dataset["text"]

dataset = standardize_sharegpt(non_reasoning_dataset)


def format_non_reasoning(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in examples["conversations"]
    ]
    return {"text": texts}


non_reasoning_dataset = dataset.map(format_non_reasoning, batched=True)
non_reasoning_conversations = non_reasoning_dataset["text"]

chat_percentage = 0.75
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations) * (1.0 - chat_percentage)),
    random_state=2407,
)

data = pd.concat([pd.Series(reasoning_conversations), pd.Series(non_reasoning_subset)])

data.name = "text"

combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed=3407)

config = SFTConfig(
    dataset_text_field="text",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=30,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    eval_dataset=None,
    args=config,
)

trainer_stats = trainer.train()


messages = [{"role": "user", "content": "What is the derivative of ln(x^3 + 1)?"}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # True
)

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

model.push_to_hub("sachinkum0009/Qwen3-4B-LoRA")
tokenizer.push_to_hub("sachinkum0009/Qwen3-4B-LoRA")
