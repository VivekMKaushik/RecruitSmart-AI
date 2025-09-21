import google.generativeai as genai

# 🔑 Directly set API Key here
GOOGLE_API_KEY = "AIzaSyC6PJg_2EJ7ngBt8D79k-CahGv6FSH3Ybs"
genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_response(input_text, pdf_text, prompt, model_name="gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([input_text, pdf_text, prompt])
    return response.text

# Add this new function for chat
def chat_with_gemini(user_message, history=[]):
    try:
        # Create model
        model = genai.GenerativeModel('gemini-pro')
        
        # Start a chat session with history
        chat = model.start_chat(history=[])
        
        # Add previous messages to context
        context = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        
        # Combine context with new message
        full_message = context + f"User: {user_message}\nAssistant:"
        
        # Get response
        response = chat.send_message(full_message)
        
        # Update history
        updated_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.text}
        ]
        
        return response.text, updated_history
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, history