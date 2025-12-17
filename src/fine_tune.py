"""
Script is used to fine tune a pre-trained llm model on a custom dataset using PEFT.
"""

import os
import numpy as np
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
import evaluate

# Force single GPU to avoid DataParallel issues
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def preprocess_function(examples, tokenizer, max_length=128):
    """Preprocess the dataset for seq2seq task."""
    inputs = examples["sentence"]
    targets = examples["text_label"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length, 
        padding="max_length", 
        truncation=True
    )
    
    labels = tokenizer(
        targets, 
        max_length=max_length, 
        padding="max_length", 
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # For logits, take argmax
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    
    # Simple accuracy metric instead of ROUGE to save memory
    # Count exact token matches
    matches = (predictions == labels).sum()
    total = labels.size
    
    return {"accuracy": matches / total}


def main():
    # Load dataset - using imdb for text classification converted to seq2seq
    print("Loading dataset...")
    dataset = load_dataset("imdb")
    
    # Take a smaller subset for faster training (only train and test splits)
    dataset = DatasetDict({
        "train": dataset["train"].select(range(1000)),
        "test": dataset["test"].select(range(200))
    })
    
    # Map sentiment labels to text for seq2seq format
    label_map = {0: "negative", 1: "positive"}
    dataset["train"] = dataset["train"].map(lambda x: {"sentence": x["text"], "text_label": label_map[x["label"]]})
    dataset["test"] = dataset["test"].map(lambda x: {"sentence": x["text"], "text_label": label_map[x["label"]]})
    
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    model_name = "bigscience/mt0-large"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Load model
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],  # Target attention layers
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output/bigscience-mt0-large-lora",
        learning_rate=1e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        logging_steps=50,
        push_to_hub=False,
        use_cpu=False,
        dataloader_num_workers=0,
        no_cuda=False,
        ddp_find_unused_parameters=False,
        fp16=True,
        include_tokens_per_second=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    model.save_pretrained("./output/bigscience-mt0-large-lora-final")
    tokenizer.save_pretrained("./output/bigscience-mt0-large-lora-final")

    
    print("Training complete!")

    model.push_to_hub("sachinkum0009/bigscience-mt0-large-lora-final")
    print("model published to hf")


if __name__ == "__main__":
    main()
