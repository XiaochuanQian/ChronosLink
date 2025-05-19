import http.client
import json
import os
import tempfile
import requests
import azure.cognitiveservices.speech as speechsdk
from pathlib import Path

# # Constants - 这些应该存储在环境变量中或安全配置中
# OPENAI_API_KEY = "sk-0EEimp6v5JbJKA4J0Lz0TrIUNzul1FeYPlgzojyTCOHFPuIV"
# OPENAI_API_URL = "api.chatanywhere.tech"
# AZURE_SPEECH_KEY = "fcde7238a41d472681f1b99a2d7acd58"  # 在实际使用时替换为您的Azure语音服务密钥
# AZURE_SPEECH_REGION = "southeastasia"  # 在实际使用时替换为您的Azure语音服务区域

# Load configuration
def load_config():
    config_path = pathlib.Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

# Load credentials from config
config = load_config()
OPENAI_API_KEY = config.get('openai', {}).get('api_key')
OPENAI_API_BASE_URL = config.get('openai', {}).get('api_base_url')
AZURE_SPEECH_KEY = config.get('azure', {}).get('speech_key')
AZURE_SPEECH_REGION = config.get('azure', {}).get('speech_region')

def text_to_speech(text, voice="alloy"):
    """
    Convert text to speech using the OpenAI TTS API
    
    Args:
        text (str): Text to convert to speech
        voice (str): Voice type (alloy, echo, fable, onyx, nova, shimmer)
        
    Returns:
        bytes: Audio data in bytes format
        bool: Success status
    """
    try:
        conn = http.client.HTTPSConnection(OPENAI_API_URL)
        
        payload = json.dumps({
            "model": "tts-1",
            "input": text,
            "voice": voice
        })
        
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        conn.request("POST", "/v1/audio/speech", payload, headers)
        response = conn.getresponse()
        
        if response.status == 200:
            audio_data = response.read()
            return audio_data, True
        else:
            print(f"TTS API error: {response.status} {response.reason}")
            return None, False
            
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return None, False

def save_audio_to_temp_file(audio_data):
    """
    Save audio data to a temporary file
    
    Args:
        audio_data (bytes): Audio data in bytes format
        
    Returns:
        str: Path to the temporary audio file
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(audio_data)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"Error saving audio to temp file: {str(e)}")
        return None

def speech_to_text(audio_file_path):
    """
    Convert speech to text using Azure Cognitive Services
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text
        bool: Success status
    """
    try:
        # Create speech configuration object
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY, 
            region=AZURE_SPEECH_REGION
        )
        
        # Create audio configuration object
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        
        # Start speech recognition
        result = speech_recognizer.recognize_once_async().get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text, True
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
            return None, False
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech recognition canceled: {cancellation_details.reason}")
            print(f"Error details: {cancellation_details.error_details}")
            return None, False
        
    except Exception as e:
        print(f"STT error: {str(e)}")
        return None, False

def cleanup_temp_file(file_path):
    """
    Clean up temporary file
    
    Args:
        file_path (str): Path to the temporary file
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning up temp file: {str(e)}") 