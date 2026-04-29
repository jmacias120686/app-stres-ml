from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API Python funcionando",
        "status": "ok"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "stress-ml-api"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)