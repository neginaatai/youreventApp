from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

app = Flask(__name__)

hf_model_name = "neginaatai/event-gpt2"

# Load tokenizer & model efficiently
try:
    tokenizer = GPT2Tokenizer.from_pretrained(hf_model_name)
    model = GPT2LMHeadModel.from_pretrained(
        hf_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route("/generate", methods=["POST"])
def generate_invitation():
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400

        inputs = tokenizer(data["prompt"], return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated_text})

    except Exception as e:
        print(f"Error generating text: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
