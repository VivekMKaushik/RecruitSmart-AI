
# llm_utils.py

import os
import google.generativeai as genai
import openai

# Choose provider from env or default
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()

# Configure API keys (read from standard env var names)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure providers defensively; don't raise if keys missing
if GOOGLE_API_KEY and GOOGLE_API_KEY != "your-api-key-here":
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        # Defer errors to runtime call sites to keep import safe
        pass

if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key-here":
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        pass

def get_response(input_text: str, pdf_text: str = "", prompt: str = "",
                 model_provider: str = MODEL_PROVIDER,
                 model_name: str = None) -> str:
    """
    Unified function to get LLM responses from Gemini, OpenAI, or HuggingFace.
    """
    model_provider = model_provider.lower()

    if model_provider == "gemini":
        model_name = model_name or "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([input_text, pdf_text, prompt])
        return response.text

    elif model_provider == "openai":
        model_name = model_name or "gpt-4o-mini"
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an HR recruiter."},
                {"role": "user", "content": f"{input_text}\n\n{pdf_text}\n\n{prompt}"}
            ]
        )
        return response["choices"][0]["message"]["content"]

    elif model_provider == "hf":
        # placeholder for HuggingFace inference API
        return hf_inference(input_text, pdf_text, prompt, model_name)

    else:
        raise ValueError(f"Unsupported MODEL_PROVIDER: {model_provider}")

# Example HuggingFace inference (stub)
def hf_inference(input_text, pdf_text, prompt, model_name="distilbert-base-uncased"):
    return f"[HF inference with {model_name}] Input: {input_text[:100]}..."

# Compatibility: add get_gemini_response for chat_utils.py and main.py
def get_gemini_response(input_text, pdf_text, prompt, model_name="gemini-1.5-flash"):
    if not GOOGLE_API_KEY:
        return "LLM not configured. Set GOOGLE_API_KEY to enable Gemini features."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([input_text, pdf_text, prompt])
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {e}"
