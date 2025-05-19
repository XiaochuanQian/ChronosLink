# ChronosLink Smart Scheduler Assistant

ChronosLink is an AI-powered calendar management assistant that helps you manage your schedule through natural language conversations. It features voice chat capabilities for a more interactive experience.

## Features

- Create, update, and delete calendar events using natural language
- Find available time slots for meetings and appointments
- View daily, weekly, or monthly schedules
- Support for multiple users with separate calendars
- Voice chat capabilities (speech-to-text and text-to-speech)
- Persistent conversation history using LangGraph's MemorySaver

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your API keys:
   - OpenAI API key for text-to-speech
   - Azure Speech Services for speech-to-text

4. Update the API keys in the `voice_utils.py` file:
   ```python
   # Replace with your actual API keys
   OPENAI_API_KEY = "your_openai_api_key"
   AZURE_SPEECH_KEY = "your_azure_speech_key"
   AZURE_SPEECH_REGION = "your_azure_region"
   ```

5. Run the application:
   ```
   python main.py
   ```

## Voice Chat

ChronosLink includes voice chat functionality:

- **Speech-to-Text**: Record your voice message using the microphone input
- **Text-to-Speech**: Enable voice output to hear the assistant's responses

To use voice features:
1. Check "Enable Voice Output" to hear the assistant's responses
2. Select a voice type from the dropdown menu
3. Use the microphone input to record your voice message
4. Click "Send Voice Message" to send your recorded message

## Multi-User Support

ChronosLink supports multiple users with separate calendars:

1. Enter a user ID in the "User ID" field
2. Click "Login" to switch to that user's calendar
3. Each user's calendar data and conversation history are stored separately

## Data Storage

Calendar data is stored locally in the `calendar_data` directory. Conversation history is maintained using LangGraph's MemorySaver for persistence across sessions. 