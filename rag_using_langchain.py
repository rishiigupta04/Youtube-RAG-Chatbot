from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from warnings import filterwarnings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
from langdetect import detect
import re
filterwarnings("ignore", category=UserWarning, module="langchain_community.vectorstores.faiss", lineno=0)



# =====================
# 1. Transcript Handling
# =====================


def extract_video_id(url_or_id: str) -> str:
    """
    Extracts the YouTube video ID from a URL or returns the ID if already provided.
    Supports common YouTube URL formats.
    """
    if re.match(r'^[\w-]{11}$', url_or_id):
        return url_or_id
    match = re.search(r'(?:v=|\/embed\/|youtu\.be\/|\/v\/|\/shorts\/)([\w-]{11})', url_or_id)
    if match:
        return match.group(1)
    match = re.search(r'([\w-]{11})', url_or_id)
    if match:
        return match.group(1)
    raise ValueError("Could not extract a valid YouTube video ID from input.")


transcript_cache = {}

def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetch transcript from a YouTube video using its ID.
    Uses cache to avoid repeated API calls for the same video.
    Returns the transcript as a single string.
    """
    cache_key = f"{video_id}:{language}"
    if cache_key in transcript_cache:
        return transcript_cache[cache_key]
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id, languages=[language])
        transcript = " ".join([snippet.text for snippet in transcript_data])
        transcript_cache[cache_key] = transcript
        return transcript
    except TranscriptsDisabled:
        print("No captions available for this video.")
        transcript_cache[cache_key] = ""
        return ""

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Returns the language code (e.g., 'en', 'hi').
    """
    try:
        return detect(text)
    except Exception:
        return "en"  # Default to English if detection fails

def translate_to_english(text: str, src_lang: str) -> str:
    """
    Translate text to English using GoogleTranslator if not already English.
    """
    if src_lang == "en":
        return text
    try:
        translated = GoogleTranslator(source=src_lang, target="en").translate(text)
        print(f"Transcript translated from {src_lang} to English.")
        return translated
    except Exception as e:
        print(f"Translation failed: {e}")
        return text



# =====================
# 2. Chunking and Text Processing
# =====================



def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 80):
    """
    Split text into smaller chunks for embedding.
    Returns a list of chunk objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.create_documents([text])
    return chunks




# =====================
# 3. Embedding & Retrieval
# =====================



def create_vector_store(chunks, embedding_model: str = "nomic-embed-text:v1.5"):
    """
    Create a FAISS vector store from text chunks using Ollama embeddings.
    Returns the vector store object.
    """
    embedding = OllamaEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding)
    return vector_store

def get_retriever(vector_store, k: int = 5):
    """
    Create a retriever to find relevant chunks for a question.
    Returns the retriever object.
    """
    retrieved_docs = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retrieved_docs

def format_docs(retrieved_docs):
    """
    Combine the text from multiple retrieved documents into a single string to be given inside prompt as context.
    """
    return "\n\n".join(doc.page_content for doc in retrieved_docs)



# =====================
# 4. Prompt & LLM
# =====================



def build_prompt():
    """
    Create a prompt template for the language model.
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant that helps people find information.\n"
            "Use the following pieces of context to answer the question at the end.\n"
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
            "{context}\n"
            "Question: {question}"
        ),
    )

def get_llm(model_name: str = "mistral:latest", temperature: float = 0.2):
    """
    Get the language model for answering questions.
    """
    return ChatOllama(model=model_name, temperature=temperature)



# =====================
# 5. Main Pipeline
# =====================



def run_chain_pipeline(url_or_id, question, chunk_size=600, chunk_overlap=80, k=3, model_name="mistral:latest", embedding_model="nomic-embed-text:v1.5", status_callback=None):
    def status(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    status("Getting transcript...")
    transcript = fetch_transcript(extract_video_id(url_or_id))
    if not transcript:
        status("Transcript not found.")
        return "Transcript not found.", "Transcript not found."
    status("Detecting transcript language...")
    lang = detect_language(transcript)
    status(f"Transcript language detected: {lang}")
    transcript = translate_to_english(transcript, lang)
    status("Splitting transcript into chunks...")
    chunks = split_text(transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    status(f"Number of chunks: {len(chunks)}")
    status("Creating vector store...")
    vector_store = create_vector_store(chunks, embedding_model=embedding_model)
    status("Setting up retriever...")
    retriever = get_retriever(vector_store, k=k)
    status("Building prompt and LLM...")
    prompt = build_prompt()
    model = get_llm(model_name=model_name)
    parser = StrOutputParser()



    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })


    sequential_chain = prompt | model | parser

    final_chain = parallel_chain | sequential_chain


    status("Running chain pipeline...")
    result = final_chain.invoke(question)
    print("\n--- Chain Pipeline Answer ---")
    print(result)
    return str(result), None

# =====================
# 6. Entry Point
# =====================

def main():
    url_or_id = "https://www.youtube.com/watch?v=OpUEEr-F5dY"  # Change this to test
    question = "Who is the major suspect? What all do we know of them?"
    run_chain_pipeline(url_or_id, question)

if __name__ == "__main__":
    main()