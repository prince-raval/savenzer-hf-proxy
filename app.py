from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os

app = Flask(__name__)
CORS(app)


HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "google/gemma-2-2b-it"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Please type a message."}), 400

        payload = {"inputs": f"User: {user_message}\nAssistant:"}
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        result = r.json()

        reply = "Sorry, no response."
        if isinstance(result, list) and "generated_text" in result[0]:
            reply = result[0]["generated_text"].split("Assistant:")[-1].strip()
        elif "error" in result:
            reply = f"Error: {result['error']}"

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"status": "Proxy running successfully!"})


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


