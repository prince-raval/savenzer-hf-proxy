from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, json

app = Flask(__name__)
CORS(app)

# ✅ WORKING MODEL
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # 💥 Guaranteed working model
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

        payload = {"inputs": f"User: {user_message}\nAssistant:"}

        print("\n=== Sending to Hugging Face ===")
        print(json.dumps(payload, indent=2))
        print("==============================")

        response = requests.post(HF_URL, headers=HEADERS, json=payload, timeout=60)

        print(f"Response Status: {response.status_code}")
        print("Response Text:", response.text[:400])

        if response.status_code == 503:
            return jsonify({
                "reply": "⏳ Model is loading on Hugging Face, please try again in 30 seconds."
            }), 503

        if response.status_code == 404:
            return jsonify({
                "reply": "⚠️ Model not found. Please verify model name or API access."
            }), 404

        if response.status_code != 200:
            return jsonify({
                "reply": f"⚠️ Error from Hugging Face ({response.status_code}): {response.text}"
            }), 502

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            reply = result[0]["generated_text"].split("Assistant:")[-1].strip()
        elif isinstance(result, dict) and "generated_text" in result:
            reply = result["generated_text"]
        else:
            reply = "⚠️ Unexpected response format."

        return jsonify({"reply": reply})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"status": "Proxy running successfully!"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
