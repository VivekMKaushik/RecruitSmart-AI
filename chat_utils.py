from gemini_utils import get_gemini_response
import prompts
import base64
import io
from PIL import Image
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------- TF-IDF Cosine Similarity ----------
def tfidf_similarity(jd_text: str, resume_text: str) -> float:
    """
    Compute cosine similarity between JD and Resume using TF-IDF.
    Returns a score between 0–100.
    """
    docs = [jd_text, resume_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)

# ---------- BM25-like Similarity (Alternative implementation) ----------
def bm25_similarity(jd_text: str, resume_text: str) -> float:
    """
    Compute BM25-like relevance score using TF-IDF with different weighting.
    Returns a score between 0–100.
    """
    docs = [jd_text, resume_text]
    vectorizer = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=True)
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    # Convert to array and normalize to 0-100 scale
    scores = tfidf_matrix.toarray()
    max_score = np.max(scores) if np.max(scores) > 0 else 1
    normalized_score = (scores[1].sum() / max_score) * 100
    
    return round(min(normalized_score, 100), 2)

def analyze_resume_vs_jd(jd_text, resume_text):
    tfidf_score = tfidf_similarity(jd_text, resume_text)
    bm25_score = bm25_similarity(jd_text, resume_text)

    return {
        "TF-IDF Score": tfidf_score,
        "BM25-like Score": bm25_score,
        "Average Score": round((tfidf_score + bm25_score) / 2, 2)
    }

# ... rest of your chat_utils.py code remains the same ...
def chat_with_ai(prompt, history, uploaded_files=None):
    """
    Adapter: build a conversation context from history, call Gemini, and
    return (response_text, updated_history_list).
    Supports images, documents, and text files.
    """
    # Build context from history
    context_parts = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        context_parts.append(f"{role.capitalize()}: {content}")
    context = "\n".join(context_parts)

    # Process uploaded files if any
    file_content = ""
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            file_name = uploaded_file.name
            
            if file_type.startswith('image/'):
                # Process image file
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file)
                    # Convert to base64 for text representation
                    buffered = io.BytesIO()
                    image_format = image.format if image.format else "PNG"
                    image.save(buffered, format=image_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    file_content += f"\n[Uploaded Image: {file_name} (base64 encoded)]"
                except Exception as e:
                    file_content += f"\n[Error processing image {file_name}: {str(e)}]"
            
            elif file_type == 'application/pdf':
                # Process PDF file
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += page_text + "\n"
                    file_content += f"\n[PDF Content from {file_name}]:\n{pdf_text}"
                except Exception as e:
                    file_content += f"\n[Error processing PDF {file_name}: {str(e)}]"
            
            elif file_type == 'text/plain':
                # Process text file
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    text_content = uploaded_file.read().decode("utf-8")
                    file_content += f"\n[Text Content from {file_name}]:\n{text_content}"
                except Exception as e:
                    file_content += f"\n[Error processing text file {file_name}: {str(e)}]"
            
            elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # Process Word document
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    doc = docx.Document(io.BytesIO(uploaded_file.read()))
                    doc_text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
                    file_content += f"\n[Word Document Content from {file_name}]:\n{doc_text}"
                except Exception as e:
                    file_content += f"\n[Error processing Word document {file_name}: {str(e)}]"
            
            else:
                file_content += f"\n[Unsupported file type: {file_name} ({file_type})]"

    # Use prompt template from prompts.py if available
    template = getattr(
        prompts,
        "chat_prompt",
        "You are a helpful assistant. Use the conversation context and any uploaded files to answer the final question.\n\nContext:\n{context}\n\nUploaded Files:\n{files}\n\nQuestion:\n{question}"
    )

    try:
        filled_prompt = template.format(context=context, files=file_content, question=prompt)
    except Exception:
        filled_prompt = f"{template}\n\nContext:\n{context}\n\nUploaded Files:\n{file_content}\n\nQuestion:\n{prompt}"

    # Call Gemini wrapper
    response_text = get_gemini_response(filled_prompt, file_content, prompt)

    # Update history and return
    updated_history = list(history)  # shallow copy
    updated_history.append({"role": "assistant", "content": response_text})
    return response_text, updated_history