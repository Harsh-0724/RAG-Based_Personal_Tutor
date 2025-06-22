# AI Tutor Backend - FastAPI Server
# pip install fastapi uvicorn python-multipart groq python-docx PyPDF2 python-dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from typing import List
from groq import Groq
import PyPDF2
from docx import Document
import io
import re
from datetime import datetime
import uvicorn
from dotenv import load_dotenv ## API KEY FIX ##

# API KEY FIX: Load environment variables from a .env file
load_dotenv()

# 1. Initialize FastAPI app
app = FastAPI(title="AI Tutor API", description="RAG-powered learning assistant")

# 2. CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. API KEY FIX: Initialize Groq client securely from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
groq_client = Groq(api_key=GROQ_API_KEY)


# 4. In-memory storage
documents_store = {}
user_sessions = {}

# 5. Pydantic Models (Data Shapes) - Unchanged
class QueryRequest(BaseModel):
    query: str
    session_id: str

class DocumentResponse(BaseModel):
    id: str
    name: str
    size: int
    upload_date: str
    chunk_count: int

class ChatResponse(BaseModel):
    response: str
    sources_used: List[str]
    session_id: str

# 6. Utility Functions - Unchanged
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def extract_text_from_file(filename: str, file_content: bytes) -> str:
    file_ext = filename.lower().split('.')[-1]
    if file_ext == 'pdf': return extract_text_from_pdf(file_content)
    elif file_ext == 'docx': return extract_text_from_docx(file_content)
    elif file_ext == 'txt':
        try: return file_content.decode('utf-8')
        except UnicodeDecodeError: return file_content.decode('utf-8', errors='ignore')
    else: raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    text = re.sub(r'\s+', ' ', text.strip())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [chunk for chunk in chunks if chunk.strip()]

def retrieve_relevant_chunks(query: str, session_id: str, top_k: int = 5) -> List[dict]:
    if session_id not in user_sessions: return []
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    if not query_words: return []
    scored_chunks = []
    for doc_id in user_sessions[session_id].get('documents', []):
        doc = documents_store.get(doc_id)
        if not doc: continue
        for i, chunk in enumerate(doc['chunks']):
            chunk_lower = chunk.lower()
            chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
            common_words = query_words.intersection(chunk_words)
            if common_words:
                score = len(common_words) / len(query_words.union(chunk_words))
                scored_chunks.append({'chunk': chunk, 'score': score, 'source': doc['name'], 'chunk_index': i})
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:top_k]

# 7. API Endpoints
@app.post("/upload", response_model=List[DocumentResponse])
async def upload_documents(files: List[UploadFile] = File(...), session_id: str = Form("default")):
    # CHAT HISTORY FIX: Initialize chat history when a new session is created
    if session_id not in user_sessions:
        user_sessions[session_id] = {'documents': [], 'chat_history': []}
    
    uploaded_docs = []
    for file in files:
        try:
            content = await file.read()
            text = extract_text_from_file(file.filename, content)
            if len(text.strip()) < 20: continue
            chunks = chunk_text(text)
            doc_id = f"{session_id}_{file.filename}_{datetime.now().timestamp()}"
            document = {'id': doc_id, 'name': file.filename, 'content': text, 'chunks': chunks, 'size': len(content), 'upload_date': datetime.now().isoformat(), 'session_id': session_id}
            documents_store[doc_id] = document
            user_sessions[session_id]['documents'].append(doc_id)
            uploaded_docs.append(DocumentResponse(id=doc_id, name=file.filename, size=len(content), upload_date=document['upload_date'], chunk_count=len(chunks)))
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")
    if not uploaded_docs: raise HTTPException(status_code=400, detail="No valid documents were processed.")
    return uploaded_docs

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: QueryRequest):
    if not request.query.strip(): raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    ## CHAT HISTORY FIX: Retrieve chat history for the current session ##
    if request.session_id not in user_sessions:
        # This is a fallback, upload should always create the session
        user_sessions[request.session_id] = {'documents': [], 'chat_history': []}
    
    chat_history = user_sessions[request.session_id].get('chat_history', [])

    try:
        # RAG retrieval remains the same, finding context for the *current* question
        relevant_chunks = retrieve_relevant_chunks(request.query, request.session_id, top_k=3) # Reduced top_k to save context space
        
        if relevant_chunks:
            context = "\n---\n".join([f"Source: {chunk['source']}\nContent: {chunk['chunk']}" for chunk in relevant_chunks])
            sources_used = sorted(list(set([chunk['source'] for chunk in relevant_chunks])))
        else:
            context = "No relevant information was found in the uploaded documents for this specific question."
            sources_used = []
        
        system_prompt = """You are an expert AI tutor. Your goal is to help a student understand their study materials. Use the provided chat history to understand the flow of the conversation. Base your answer for the user's latest question *strictly* on the provided context from their documents. If the context doesn't contain the answer, state that explicitly. Do not make up information."""
        
        # CHAT HISTORY FIX: Construct the messages list with history
        messages_to_send = [{"role": "system", "content": system_prompt}]
        
        # Add past interactions, but limit to save tokens (e.g., last 5 turns/10 messages)
        messages_to_send.extend(chat_history[-10:])
        
        # Add the new user query with its RAG context
        user_prompt_with_context = f"Context from documents:\n{context}\n\nQuestion: {request.query}"
        messages_to_send.append({"role": "user", "content": user_prompt_with_context})
        
        chat_completion = groq_client.chat.completions.create(
            messages=messages_to_send,
            model="llama3-70b-8192",
            max_tokens=1500,
            temperature=0.7
        )
        response_text = chat_completion.choices[0].message.content
        
        # CHAT HISTORY FIX: Update the chat history with the new turn
        # Store the original user query (without the bulky context)
        user_sessions[request.session_id]['chat_history'].append({"role": "user", "content": request.query})
        user_sessions[request.session_id]['chat_history'].append({"role": "assistant", "content": response_text})
        
        return ChatResponse(response=response_text, sources_used=sources_used, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the AI response: {str(e)}")

@app.get("/documents/{session_id}", response_model=List[DocumentResponse])
async def get_documents(session_id: str):
    if session_id not in user_sessions: return []
    doc_ids = user_sessions[session_id].get('documents', [])
    doc_list = []
    for doc_id in doc_ids:
        if doc_id in documents_store:
            doc = documents_store[doc_id]
            doc_list.append(DocumentResponse(id=doc['id'], name=doc['name'], size=doc['size'], upload_date=doc['upload_date'], chunk_count=len(doc['chunks'])))
    return doc_list

@app.delete("/documents/{session_id}/{doc_id}")
async def delete_document(session_id: str, doc_id: str):
    if session_id in user_sessions and doc_id in user_sessions[session_id]['documents']:
        user_sessions[session_id]['documents'].remove(doc_id)
        if doc_id in documents_store: del documents_store[doc_id]
        # Also clear chat history if a document is deleted, as context has changed
        user_sessions[session_id]['chat_history'] = []
        return {"message": "Document deleted successfully"}
    raise HTTPException(status_code=404, detail="Document not found")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "total_documents": len(documents_store), "active_sessions": len(user_sessions)}


# 8. Mount the Static Files Directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# 9. Entry point for running the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)