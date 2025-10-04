# Youtube Q&A Chatbot

This is a Generative AI chatbot that answers questions from YouTube videos using:
- **Gemini API** for embeddings and response generation
- **Pinecone** for vector search
- **Faster Whisper** for transcription
- **Streamlit** for the app interface
- **gTTS** and **SpeechRecognition** for voice I/O

**Built:** June 2025  
**Updated for portfolio:** October 2025  

## Features
- Transcribes any YouTube video (no length limits)
- Caches transcripts for faster use
- Interactive Q&A with voice support

## Setup
1. Clone the repo
2. Add your `.env` file with API keys
3. Run:
   ```bash
   streamlit run app2.py
   ```
