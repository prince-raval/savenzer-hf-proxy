from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, json

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta" 
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
        print("Response Text:", response.text[:400])  # preview of API response

        # If Hugging Face API is still loading the model
        if response.status_code == 503:
            return jsonify({
                "reply": "⏳ The model is loading on Hugging Face. Please try again in a few seconds."
            }), 503

        # If Hugging Face gives any other error
        if response.status_code != 200:
            return jsonify({
                "reply": f"⚠️ Hugging Face error ({response.status_code}): {response.text}"
            }), 502

        try:
            result = response.json()
        except json.JSONDecodeError:
            return jsonify({"reply": "⚠️ Invalid response from Hugging Face."}), 502

        # Extract model output
        if isinstance(result, list) and "generated_text" in result[0]:
            reply = result[0]["generated_text"].split("Assistant:")[-1].strip()
        elif "error" in result:
            reply = f"⚠️ Model Error: {result['error']}"
        else:
            reply = "⚠️ No valid output received from model."

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

