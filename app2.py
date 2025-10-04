import os
import hashlib
from faster_whisper import WhisperModel
from pydub import AudioSegment
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
import tempfile
import time
from tqdm import tqdm
import yt_dlp
from dotenv import load_dotenv

# ========== LOAD ENV ==========
load_dotenv()

# ========== CONFIG ==========
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "youtube-gemini")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini Setup
genai.configure(api_key=GOOGLE_API_KEY)
generation_model = genai.GenerativeModel("gemini-1.5-flash")

# Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    time.sleep(2)
index = pc.index(PINECONE_INDEX)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ========== HELPERS ==========
def get_video_id(url):
    import re
    match = re.search(r"(?:v=|youtu.be/)([\w-]+)", url)
    return match.group(1) if match else hashlib.md5(url.encode()).hexdigest()

def download_audio(url, mp3_path):
    temp_download_path = mp3_path.replace(".mp3", ".webm")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_download_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Convert to mp3 with pydub
    audio = AudioSegment.from_file(temp_download_path)
    audio.export(mp3_path, format="mp3")

    # 🧹 Clean up temp file
    if os.path.exists(temp_download_path):
        os.remove(temp_download_path)

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

def transcribe_audio(wav_path):
    model = WhisperModel("base", compute_type="int8", device="cpu")
    segments, _ = model.transcribe(wav_path)
    return " ".join([seg.text for seg in segments])

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embed_text(text):
    res = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")
    return res["embedding"]

def upload_chunks_to_pinecone(chunks, namespace):
    vectors = []
    for i, chunk in enumerate(tqdm(chunks)):
        try:
            emb = embed_text(chunk)
            vectors.append({"id": f"chunk-{i}", "values": emb, "metadata": {"text": chunk}})
        except Exception as e:
            print(f"Chunk {i} failed: {e}")
    index.upsert(vectors=vectors, namespace=namespace)

def search_context(query, namespace):
    query_emb = embed_text(query)
    res = index.query(vector=query_emb, top_k=3, include_metadata=True, namespace=namespace)
    return "\n".join([m["metadata"]["text"] for m in res["matches"]])

def answer_question(question, namespace):
    context = search_context(question, namespace)
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    return generation_model.generate_content(prompt).text

def listen_to_voice():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            st.warning("❌ Could not understand your voice input.")
            return ""
        except sr.RequestError:
            st.warning("⚠️ Voice recognition service unavailable.")
            return ""
    except OSError:
        st.warning("🎙️ Voice input not available in this environment.")
        return ""

def speak(text):
    tts = gTTS(text)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="🎥 YouTube Q&A Bot", layout="centered")
st.title("🎥 YouTube Q&A Chatbot (Gemini + Pinecone)")

url = st.text_input("Paste a YouTube video URL")
process_btn = st.button("🔁 Process Video")

if url and process_btn:
    video_id = get_video_id(url)
    cache_path = os.path.join(CACHE_DIR, f"{video_id}.txt")

    if os.path.exists(cache_path):
        st.success("✅ Using cached transcript")
        with open(cache_path, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
        with st.spinner("🎧 Downloading audio and transcribing..."):
            with tempfile.TemporaryDirectory() as tmp:
                mp3_path = os.path.join(tmp, "audio.mp3")
                wav_path = os.path.join(tmp, "audio.wav")
                download_audio(url, mp3_path)
                convert_to_wav(mp3_path, wav_path)
                transcript = transcribe_audio(wav_path)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(transcript)
        st.success("✅ Transcript ready")

    chunks = chunk_text(transcript)
    with st.spinner("📤 Uploading to Pinecone..."):
        try:
            index.delete(delete_all=True, namespace=video_id)
        except:
            pass
        upload_chunks_to_pinecone(chunks, namespace=video_id)
    st.success("✅ Video indexed and ready!")
    st.session_state.indexed = True

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if url and not process_btn:
    video_id = get_video_id(url)
    if os.path.exists(os.path.join(CACHE_DIR, f"{video_id}.txt")):
        st.session_state.indexed = True

if st.session_state.indexed or process_btn:
    st.subheader("💬 Ask a question")
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Your question")
    with col2:
        voice = st.button("🎙️ Speak")
        if voice:
            question = listen_to_voice()
            st.write(f"🗣️ You said: **{question}**")

    if st.button("Submit") and question:
        with st.spinner("🤖 Thinking..."):
            answer = answer_question(question, namespace=get_video_id(url))
            st.write("**🤖 Gemini says:**")
            st.success(answer)
            st.audio(speak(answer), format="audio/mp3")
            st.write("✅ Answer generated and spoken!")