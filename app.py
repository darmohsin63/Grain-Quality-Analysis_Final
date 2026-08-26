from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from src.image_processing import analyze_grain_image

app = Flask(__name__)
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("grain_image")
    if uploaded is None or uploaded.filename == "":
        return render_template("index.html", error="Please select a grain image.")

    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    extension = os.path.splitext(uploaded.filename)[1].lower()
    if extension not in allowed:
        return render_template("index.html", error="Please upload JPG, JPEG, PNG or WEBP.")

    job_id = uuid.uuid4().hex[:10]
    filename = secure_filename(uploaded.filename)
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")
    uploaded.save(input_path)

    try:
        result = analyze_grain_image(input_path, OUTPUT_DIR, job_id)
    except Exception as exc:
        return render_template("index.html", error=f"Image processing failed: {exc}")

    return render_template("results.html", result=result)

@app.route("/files/<filename>")
def files(filename):
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
