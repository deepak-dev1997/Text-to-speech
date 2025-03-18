from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import json
import subprocess
from vosk import Model, KaldiRecognizer
import pyttsx3
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the larger Vosk model
vosk_model = Model("model/vosk-model-en-us-0.42-gigaspeech")
# Create a recognizer for 16kHz PCM audio
recognizer = KaldiRecognizer(vosk_model, 16000)

# Initialize pyttsx3 TTS engine
tts_engine = pyttsx3.init()

def convert_audio(audio_bytes):
    """
    Convert input audio from WebM/Opus to raw PCM (16kHz, mono)
    using ffmpeg. Ensure ffmpeg is installed on your system.
    """
    try:
        process = subprocess.Popen(
            [r'D:\Sunbase\speech to text model\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe',
             '-loglevel', 'quiet', '-i', 'pipe:0', '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, _ = process.communicate(audio_bytes)
        return out
    except Exception as e:
        print("Error converting audio:", e)
        return None

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    try:
        print("Received audio chunk")
        # Decode base64 audio chunk from the client
        audio_data = base64.b64decode(data)
        print("Audio data length:", len(audio_data))
        
        # Convert the audio to raw PCM format expected by Vosk
        pcm_data = convert_audio(audio_data)
        if pcm_data:
            print("PCM data length:", len(pcm_data))
        else:
            print("No PCM data returned from conversion.")
            return

        # Feed the PCM audio to the recognizer
        if recognizer.AcceptWaveform(pcm_data):
            result = recognizer.Result()
            print("Final result:", result)
            result_dict = json.loads(result)
            text = result_dict.get("text", "")
            if text:
                # Emit the text transcription
                emit("transcription", {"text": text}, broadcast=True)

                # Convert text to speech using pyttsx3:
                tts_filename = "tts_output.wav"
                tts_engine.save_to_file(text, tts_filename)
                tts_engine.runAndWait()

                # Read the generated audio file and encode it
                with open(tts_filename, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                # Emit the TTS audio data back to the client
                emit("tts_audio", {"audio": audio_b64}, broadcast=True)

                # Optionally, remove the temporary file
                os.remove(tts_filename)
            # Reset the recognizer for the next utterance.
            recognizer.Reset()
        else:
            partial_result = recognizer.PartialResult()  # Use PartialResult for intermediate output
            print("Partial result:", partial_result)
            result_dict = json.loads(partial_result)
            text = result_dict.get("partial", "")
            if text:
                emit("transcription", {"text": text}, broadcast=True)

                # Convert text to speech using pyttsx3:
                tts_filename = "tts_output.wav"
                tts_engine.save_to_file(text, tts_filename)
                tts_engine.runAndWait()

                # Read the generated audio file and encode it
                with open(tts_filename, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                # Emit the TTS audio data back to the client
                emit("tts_audio", {"audio": audio_b64}, broadcast=True)

                # Optionally, remove the temporary file
                os.remove(tts_filename)
                
    except Exception as e:
        print("Error processing audio chunk:", e)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    # Run the Flask-SocketIO server on port 5001
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
