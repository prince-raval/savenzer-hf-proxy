from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, traceback

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (for your website frontend)

# ==============================
# 🔧 Hugging Face Configuration
# ==============================
HF_API_KEY = os.getenv("HF_API_KEY")  # Loaded from Render environment
HF_MODEL = "google/gemma-2-2b-it"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# ✅ Log that API key loaded correctly
if HF_API_KEY:
    print("✅ Hugging Face key loaded:", HF_API_KEY[:8] + "********")
else:
    print("⚠️ Hugging Face API key not found! Check Render → Environment tab.")


# ==============================
# 💬 Chat Endpoint
# ==============================
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        # Get message from frontend
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "⚠️ Please type a message."}), 400

        # Prepare request payload for Hugging Face model
        payload = {"inputs": f"User: {user_message}\nAssistant:"}
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }

        # Send request to Hugging Face model
        response = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        result = response.json()

        print("🔹 HF Raw Response:", result)  # Log for debugging

        # Parse the response
        reply = "Sorry, no valid response from AI."

        if isinstance(result, list) and "generated_text" in result[0]:
            reply = result[0]["generated_text"].split("Assistant:")[-1].strip()
        elif "generated_text" in result:
            reply = result["generated_text"].split("Assistant:")[-1].strip()
        elif "error" in result:
            reply = f"⚠️ HF Model Error: {result['error']}"

        return jsonify({"reply": reply})

    except requests.exceptions.Timeout:
        return jsonify({"reply": "⏳ Model took too long to respond. Please try again."}), 504

    except Exception as e:
        print("❌ Server Error:", str(e))
        traceback.print_exc()  # Print full error to Render logs
        return jsonify({"reply": f"⚠️ Internal Server Error: {str(e)}"}), 500


# ==============================
# 🏠 Home Endpoint (for Render test)
# ==============================
@app.route("/")
def home():
    return jsonify({"status": "Proxy running successfully!"})


# ==============================
# 🚀 Main entry point
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
