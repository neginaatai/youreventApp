from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re

# Load fine-tuned model and tokenizer
model_path = "./gpt2-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Prepare prompt
prompt_text = "Birthday invitation for a fun summer party:"

# Encode prompt and create attention mask
inputs = tokenizer.encode(prompt_text, return_tensors="pt")
attention_mask = torch.ones_like(inputs)

# Generate multiple outputs with tuned parameters
outputs = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_length=60,           # shorter max length for concise invites
    temperature=0.7,         # less randomness for coherent text
    top_k=40,                # restrict token sampling to top_k
    top_p=0.9,               # nucleus sampling threshold
    num_return_sequences=3,  # generate 3 variations
    pad_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size=2,
    do_sample=True
)

# Decode, clean, and print generated invitations
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)

    # Truncate after first period or newline for neatness
    match = re.search(r'[\.\n]', text)
    if match:
        text = text[:match.end()]

    print(f"\nInvitation Option {i+1}:\n{text}")
