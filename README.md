# YouTube RAG Chatbot: AI-Powered Video Q&A Platform 🚀

## Overview
Unlock the knowledge in any YouTube video with our state-of-the-art Retrieval-Augmented Generation (RAG) chatbot. Powered by advanced LLMs and semantic search, this solution transforms video transcripts into an interactive, timestamp-cited Q&A experience for your users.

---

## ✨ Key Features

- **Instant Answers from Any YouTube Video**
  - Users ask questions about a video; the chatbot finds and cites relevant moments with precise timestamps.
- **Multilingual Support**
  - Automatic transcript language detection and translation to English for seamless global coverage.
- **Semantic Search & Contextual Retrieval**
  - Chunks video transcripts and indexes them with high-performance embeddings for accurate, context-rich answers.
- **Timestamped Citations**
  - Every answer references the exact video moment, boosting trust and transparency.
- **LLM-Powered Summaries**
  - Summarizes key points and source timestamps, making information easy to verify and explore further.
- **Scalable & Modular**
  - Built with LangChain, FAISS, and Ollama for flexible deployment and easy integration into your SaaS, web, or mobile platform.

---

## How It Works
1. **Transcript Extraction**: Fetches and translates YouTube video transcripts.
2. **Chunking & Embedding**: Splits transcripts into context-preserving chunks, embeds them for semantic search.
3. **Retrieval & Prompting**: Finds the most relevant transcript segments for any user question.
4. **LLM Answer Generation**: Uses a custom prompt to generate answers, always citing timestamps and summarizing sources.

---

## Use Cases
- **EdTech**: Let students ask questions about lectures and get timestamped answers.
- **Market Research**: Extract insights from interviews, podcasts, or webinars.
- **Content Discovery**: Help users find key moments in long-form videos instantly.
- **Customer Support**: Automate FAQ extraction from product demo videos.

---

## Why Choose Us?
- **Accuracy**: Answers are always grounded in the source video, with verifiable timestamps.
- **Speed**: Real-time Q&A, even for long videos.
- **Flexibility**: Integrate with your stack, customize models, and scale as you grow.

---

## Models & Infrastructure

- **Large Language Models (LLMs):**
  - Supports open-source and commercial LLMs for answer generation, including models from Hugging Face and integration with cloud APIs.
  - Easily switch between models to balance cost, privacy, and performance.
- **Embedding Models:**
  - Uses high-quality embedding models for semantic search, including local and cloud-based options.
  - Optimized for fast, accurate retrieval of relevant transcript chunks.
- **Ollama Integration:**
  - Ollama enables secure, local deployment of LLMs, keeping your data private and reducing latency.
  - Run models on your own hardware or scale in the cloud as needed.
  - Flexible architecture allows you to use Ollama, Hugging Face, or other providers with minimal configuration.

---

## Get Started

### Setup Steps
1. **Clone the Repository**
   - Open your terminal and run:
     ```cmd
     git clone https://github.com/yourorg/youtube-rag-chatbot.git
     ```
2. **Navigate to the Project Directory**
   - ```cmd
     cd "YouTube Chatbot"
     ```
3. **Create a Virtual Environment (Recommended)**
   - ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
4. **Install Dependencies**
   - ```cmd
     pip install -r requirements.txt
     ```
5. **Download & Configure Models with Ollama**
   - Install Ollama from [ollama.com](https://ollama.com/download).
   - Download the required models:
     ```cmd
     ollama pull mistral:latest
     ollama pull nomic-embed-text:v1.5
     ```
   - **Important:** Ollama must be running as a background service before launching the app. If it does not start automatically, run:
     ```cmd
     ollama serve
     ```
   - You may use other supported models by changing the model names in the code if desired.
6. **Run the Application**
   - Ensure Ollama is running in the background.
   - ```cmd
     python rag_using_langchain.py
     ```
7. **Try a Demo**
   - Follow the prompts to enter a YouTube video link and your question.

---

*Empower your users to ask, discover, and learn from any video—instantly.*
