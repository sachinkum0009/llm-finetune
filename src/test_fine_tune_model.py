# Load model directly
from transformers import AutoModel
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def main():
    model = AutoModelForSeq2SeqLM.from_pretrained("sachinkum0009/bigscience-mt0-large-lora-final")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

    model = model.to("cuda")
    model.eval()
    inputs = tokenizer("Preheat the oven to 350 degrees and place the ", return_tensors="pt")

    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

    # "Preheat the oven to 350 degrees and place the cookie dough in the center of the oven. In a large bowl, combine the flour, baking powder, baking soda, salt, and cinnamon. In a separate bowl, combine the egg yolks, sugar, and vanilla."
    

if __name__ == '__main__':
    main()

