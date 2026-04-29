from flask import Flask, jsonify
import os

app = Flask(__name__)

print("BOOT: app.py cargado", flush=True)

@app.route("/", methods=["GET"])
def home():
    print("REQUEST: /", flush=True)
    return jsonify({
        "message": "API Python funcionando",
        "status": "ok"
    })

@app.route("/health", methods=["GET"])
def health():
    print("REQUEST: /health", flush=True)
    return jsonify({
        "status": "ok"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"BOOT: iniciando Flask en puerto {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)