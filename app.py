import os
import uuid
from flask import Flask, request, jsonify
import whisper
import whisperx
import torch

# os.environ["FFMPEG_BINARY"] = r'D:\Sunbase\speech to text model\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe'
# os.environ["PATH"] += os.pathsep + r'D:\Sunbase\speech to text model\ffmpeg-master-latest-win64-gpl-shared\bin'


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print("Using device:", device)

# Load Whisper model for transcription
whisper_model = whisper.load_model("base", device=device)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.mp3'):
        return jsonify({"error": "Only MP3 files are allowed"}), 400

    os.makedirs("uploads", exist_ok=True)
    filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    try:
        # Transcription with Whisper
        result = whisper_model.transcribe(file_path)
        transcription = result.get("text", "")
        language = result.get("language", "en")

        # Load audio with WhisperX utility
        audio = whisperx.load_audio(file_path)

        # Alignment (no auth required)
        align_model, metadata = whisperx.load_align_model(language, device)
        aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, device)

        segments = []
        for seg in aligned_result["segments"]:
            segments.append({
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "")
            })

        response = {
            "transcription": transcription,
            "segments": segments
        }

    except Exception as e:
        response = {"error": str(e)}
    finally:
        os.remove(file_path)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
