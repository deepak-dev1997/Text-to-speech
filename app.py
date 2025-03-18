import os

# Set the environment variable for ffmpeg and add its directory to PATH
os.environ["FFMPEG_BINARY"] = r'D:\Sunbase\speech to text model\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe'
os.environ["PATH"] += os.pathsep + r'D:\Sunbase\speech to text model\ffmpeg-master-latest-win64-gpl-shared\bin'

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import subprocess
import pyttsx3
import tempfile
import whisper

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the Whisper model - adjust the model size as needed
whisper_model = whisper.load_model("base")

# Initialize pyttsx3 TTS engine
tts_engine = pyttsx3.init()

def convert_audio_to_wav(audio_bytes):
    """
    Convert input audio from WebM/Opus to WAV format (16kHz, mono)
    using ffmpeg.
    """
    try:
        ffmpeg_executable = os.environ.get("FFMPEG_BINARY")
        print(ffmpeg_executable)
        process = subprocess.Popen(
            [ffmpeg_executable,
             '-loglevel', 'quiet',
             '-i', 'pipe:0',
             '-ar', '16000',
             '-ac', '1',
             '-f', 'wav',
             'pipe:1'],
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
        
        # Convert the audio to WAV format (16kHz, mono)
        wav_data = convert_audio_to_wav(audio_data)
        if not wav_data:
            print("No WAV data returned from conversion.")
            return
        print("WAV data length:", len(wav_data))
        
        # Write the WAV data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(wav_data)
            temp_wav.flush()
            temp_filename = temp_wav.name
        
        # Transcribe the temporary WAV file using Whisper
        result = whisper_model.transcribe(temp_filename)
        text = result.get("text", "").strip()
        print("Transcription result:", text)
        
        # Remove the temporary audio file
        os.remove(temp_filename)
        
        if text:
            # Emit the text transcription to the client
            emit("transcription", {"text": text}, broadcast=True)
            
            # Convert the transcription back to speech using pyttsx3
            tts_filename = "tts_output.wav"
            tts_engine.save_to_file(text, tts_filename)
            tts_engine.runAndWait()
            
            # Read and encode the generated TTS audio file
            with open(tts_filename, "rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            # Emit the TTS audio data back to the client
            emit("tts_audio", {"audio": audio_b64}, broadcast=True)
            
            # Remove the temporary TTS file
            os.remove(tts_filename)
                
    except Exception as e:
        print("Error processing audio chunk:", e)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    # Run the Flask-SocketIO server on port 5001
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
