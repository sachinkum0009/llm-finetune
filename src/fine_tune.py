"""
Script is used to fine tune a pre-trained llm model on a custom dataset using PEFT.
"""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM


def main():
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


if __name__ == "__main__":
    main()
