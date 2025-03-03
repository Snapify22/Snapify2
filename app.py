import os
import time
import cv2
import easyocr
import numpy as np
import base64
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, send_file, url_for
from deep_translator import GoogleTranslator, single_detection as detect_language
from gtts import gTTS
from pydub import AudioSegment

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize OCR Reader
reader = easyocr.Reader(['en'])

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg')
    cv2.imwrite(preprocessed_path, binary)
    return preprocessed_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        preprocessed_path = preprocess_image(file_path)
        result = reader.readtext(preprocessed_path)
        extracted_text = " ".join([detection[1] for detection in result])

        return jsonify({"text": extracted_text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)
        if os.exists(preprocessed_path):
            os.remove(preprocessed_path)

@app.route('/detect_language', methods=['POST'])
def detect_text_language():
    text = request.json.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        detected_lang = detect_language(text)
        return jsonify({"detected_language": detected_lang})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target_lang", "en")

    try:
        translated_text = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/live_ocr', methods=['POST'])
def live_ocr():
    image_data = request.json.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        result = reader.readtext(img)
        detected_text = " ".join([detection[1] for detection in result])

        return jsonify({"text": detected_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    filename = f"{int(time.time())}_{audio_file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    # Convert MP3 to WAV if necessary
    if file_path.endswith('.mp3'):
        sound = AudioSegment.from_mp3(file_path)
        wav_path = file_path.replace('.mp3', '.wav')
        sound.export(wav_path, format="wav")
        file_path = wav_path

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Google Speech Recognition could not understand audio"}), 500
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)
        if file_path.endswith('.wav'):
            os.remove(file_path)

@app.route('/capture_image', methods=['POST'])
def capture_image():
    image_data = request.json.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filename = f"{int(time.time())}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(file_path, img)

        preprocessed_path = preprocess_image(file_path)
        result = reader.readtext(preprocessed_path)
        extracted_text = " ".join([detection[1] for detection in result])

        return jsonify({"text": extracted_text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    text = request.json.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        tts = gTTS(text)
        filename = f"{int(time.time())}.mp3"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        tts.save(file_path)
        return jsonify({"audio_url": url_for('static', filename=filename)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
