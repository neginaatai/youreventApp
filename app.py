from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
hf_model_name = "neginaatai/event-gpt2"
try:
    tokenizer = GPT2Tokenizer.from_pretrained(hf_model_name, cache_dir="./model_cache")
    model = GPT2LMHeadModel.from_pretrained(hf_model_name, cache_dir="./model_cache")
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

        prompt = data["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")

        # Ensure pad token is set
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
    app.run(host="127.0.0.1", port=5000, debug=True)