import os
import json
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load Ambuj's resume data
# Ensure ambuj_resume.json is in the same directory as app.py
def load_resume_data(file_path='ambuj_resume.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the resume JSON file is in the correct location.")
        # Return an empty dict if file not found, LLM will handle "not available"
        return {} 
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Please check your JSON format.")
        # Return an empty dict if JSON is invalid
        return {}

RESUME_DATA = load_resume_data()

# Define the LLM model to use from OpenRouter
# Using the stable Llama 3.3 70B Instruct model, which is available in OpenRouter's free tier
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free" # Corrected model name as per Ambuj's input
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

# Abusive words list (can be expanded)
ABUSIVE_WORDS = ["fuck", "shit", "bitch", "asshole", "damn", "idiot", "stupid", "bloody"] # Add more as needed

# --- Helper Functions ---

def is_abusive(text):
    text_lower = text.lower()
    for word in ABUSIVE_WORDS:
        if word in text_lower:
            return True
    return False

def get_llm_response(user_message, history=None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable."

    # Construct the system message with resume data and instructions for the LLM
    # This is where prompt engineering and RAG (simplified) happens
    system_message = f"""
    You are Ambuj Kumar Tripathi's personal AI assistant, built by him to answer questions about his professional background and skills based on his resume.
    You can also answer general knowledge questions, acting as a helpful AI.

    **Ambuj Kumar Tripathi's Resume Data (if available):**
    {json.dumps(RESUME_DATA, indent=2, ensure_ascii=False) if RESUME_DATA else "Resume data is not available or is empty."}

    **Key Instructions:**
    - **Always refer to Ambuj Kumar Tripathi by his full name, "Ambuj Kumar Tripathi", or just "Ambuj".**
    - **Prioritize answering questions based on the provided resume data.** If the resume data is empty, mention that you don't have Ambuj's specific resume details.
    - **If a question is clearly not related to Ambuj Kumar Tripathi's resume (e.g., general knowledge, personal opinions), you may answer it to the best of your ability as a helpful AI, but always revert to focusing on Ambuj's professional profile when possible.**
    - **If a question is related to Ambuj's profile but the specific information is NOT in the provided resume data, politely state that the information is not explicitly available.** Use this exact phrasing: "I apologize, that information is not explicitly available in the resume data Ambuj Kumar Tripathi has provided me. I am Ambuj Kumar Tripathi's AI assistant and I am still learning to retrieve more specific details."
    - **Maintain a helpful, professional, and respectful tone.**
    - **Respond in the same language as the user's query (Hindi, English, or Hinglish).** Ensure your Hindi/Hinglish responses are natural and grammatically correct.
    - **Keep answers concise and to the point, while being informative.**
    - **Do not engage in any conversational topics outside of Ambuj's resume details or general helpful AI responses.**
    - If asked to introduce yourself, state: "Hello! I am Ambuj Kumar Tripathi's AI assistant. I was created by Ambuj Kumar Tripathi to help answer questions about his professional background."
    """

    messages = [
        {"role": "system", "content": system_message}
    ]
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "stream": False, # We don't need streaming for this setup
        "temperature": 0.7 # Adjust for creativity (lower for more factual, higher for more creative)
    }

    try:
        response = requests.post(OPENROUTER_API_BASE, headers=headers, json=data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()

        # Check for specific error messages from OpenRouter
        if "error" in response_json:
            error_message = response_json["error"].get("message", "An unknown error occurred with the LLM API.")
            if "insufficient_quota" in error_message or "rate limit" in error_message:
                 return "I'm sorry, I've hit a temporary usage limit. Please try again in a few minutes or tomorrow. Ambuj Kumar Tripathi will resolve this soon."
            return f"LLM API Error: {error_message}"

        return response_json["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Network Error: Could not connect to the LLM API. Please check your internet connection. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred during LLM response generation: {e}"

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    chat_history = request.json.get('history', []) # Get existing history from frontend

    if not user_message:
        return jsonify({"response": "Please type a message."})

    # Check for abusive language first
    if is_abusive(user_message):
        return jsonify({"response": "I apologize, but I am programmed to be respectful. Please use polite language."})

    # Get response from LLM
    bot_response = get_llm_response(user_message, history=chat_history)

    return jsonify({"response": bot_response})

if __name__ == '__main__':
    # In a production environment (like Render), Flask's development server is not used.
    # Gunicorn or similar WSGI server is used instead.
    # For local testing in Cloud Shell:
    app.run(debug=True, host='0.0.0.0', port=8080)
