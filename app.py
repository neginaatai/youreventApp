from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

app = Flask(__name__)

# Use smaller model to save memory
HF_MODEL_NAME = "distilgpt2"

# Lazy-load cache
model_cache = {}

def get_model():
    if "model" not in model_cache:
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(HF_MODEL_NAME)
            model = GPT2LMHeadModel.from_pretrained(
                HF_MODEL_NAME,
                device_map={"": "cpu"},  # CPU only
                torch_dtype=torch.float16,  # half precision
                low_cpu_mem_usage=True
            )
            model.eval()
            model_cache["tokenizer"] = tokenizer
            model_cache["model"] = model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return model_cache["tokenizer"], model_cache["model"]

@app.route("/generate", methods=["POST"])
def generate_invitation():
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400

        prompt = data["prompt"]
        tokenizer, model = get_model()

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=100,  # reduced for memory
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
