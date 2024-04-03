from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

app = Flask(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="LLM API Server")
parser.add_argument("--model", type=str, default="your_model_name_or_path", help="Name or path of the model to load")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to run the server")
parser.add_argument("--port", type=int, default=5000, help="Port number to run the server")
args = parser.parse_args()

# Load the model and tokenizer
model_name = args.model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    inputs = data.get("inputs", "")
    parameters = data.get("parameters", {})

    max_new_tokens = parameters.get("max_new_tokens", 50)
    temperature = parameters.get("temperature", 1.0)

    input_ids = tokenizer.encode(inputs, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    response = [{"generated_text": generated_text}]
    return jsonify(response)

if __name__ == "__main__":
    app.run(host=args.host, port=args.port)
