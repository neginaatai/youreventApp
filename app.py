from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re

app = Flask(__name__)

# Load model and tokenizer once at startup
#replace with the new one t hugging face repo name
hf_model_name = "neginaatai/event-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_name="./model_cache")
model = GPT2LMHeadModel.from_pretrained(hf_model_name"./model_cache")

@app.route('/generate-invitation', methods=['POST'])
def generate_invitation():
    data = request.json
    prompt_text = data.get('prompt', '')

    if not prompt_text:
        return jsonify({"error": "Prompt text is required"}), 400

    # Encode prompt and create attention mask
    inputs = tokenizer.encode(prompt_text, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)

    # Generate multiple invitation options
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=60,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        num_return_sequences=3,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True
    )

    invitations = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)

        # Truncate after first period or newline
        match = re.search(r'[\.\n]', text)
        if match:
            text = text[:match.end()]

        invitations.append(text)

    return jsonify({"invitations": invitations})
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract fields Spring Boot sends (example: event type, mood, etc.)
    event_type = data.get('eventType', 'Birthday')
    mood = data.get('mood', 'Fun')

    # Example logic: map event type/mood to a template name
    template_mapping = {
        "Birthday": {"Fun": "fun_birthday_template", "Cozy": "cozy_birthday_template"},
        "Wedding": {"Romantic": "romantic_wedding_template"}
        # Add more mappings for your events
    }

    template = template_mapping.get(event_type, {}).get(mood, "default_template")

    return jsonify({"template": template})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
