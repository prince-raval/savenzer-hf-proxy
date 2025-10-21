from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, json

app = Flask(__name__)
CORS(app)

# ✅ Hugging Face Model - 100% working public chatbot
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "facebook/blenderbot-400M-distill"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please type a message."}), 400

        payload = {"inputs": user_message}

        print("\n=== Sending to Hugging Face ===")
        print(json.dumps(payload, indent=2))
        print("==============================")

        response = requests.post(HF_URL, headers=HEADERS, json=payload, timeout=60)
        print(f"Response Status: {response.status_code}")
        print("Response Text:", response.text[:400])

        if response.status_code == 503:
            return jsonify({
                "reply": "⏳ Model is loading on Hugging Face, please wait 15 seconds and try again."
            }), 503

        if response.status_code == 404:
            return jsonify({
                "reply": "⚠️ Model not found — please check Hugging Face model name."
            }), 404

        if response.status_code != 200:
            return jsonify({
                "reply": f"⚠️ Hugging Face Error ({response.status_code}): {response.text}"
            }), 502

        result = response.json()

        # The model returns a dict with "generated_text"
        reply = result[0].get("generated_text", "⚠️ No reply generated.")

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"Internal Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"status": "Proxy running successfully!"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
