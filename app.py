from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from functools import wraps
import time

# --- 1. FIREBASE INITIALIZATION AND DATA LOADING ---

# Load the secret key (assuming 'firebase_key.json' is in the root directory)
try:
    # 1. Load the secret key you downloaded
    cred = credentials.Certificate("firebase_key.json")

    # 2. Connect to Firebase
    firebase_admin.initialize_app(cred)

    # 3. Get a reference to your database (assuming you use Firestore)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}. Ensure 'firebase_key.json' is correct.")
    # Set db to None so we can check if initialization failed later
    db = None 

# NOTE: The original import below is removed as you are using google.generativeai directly.
# from resolve_bot.core import get_chatbot_response

def load_chatbot_knowledge():
    """
    Fetches the data from Firestore, formats it, and returns it as a string 
    to be used as context for the Gemini model.
    """
    if not db:
        return "Knowledge Base: Failed to load external data (Firebase initialization failed)."

    knowledge_text = "KNOWLEDGE BASE DATA FOR CHATBOT:\n"
    
    # >>> CRITICAL: Change 'support_documents' to the actual name of your Firestore collection
    COLLECTION_NAME = 'support_documents' 
    
    try:
        # Stream all documents from your knowledge collection
        docs_ref = db.collection(COLLECTION_NAME).stream()
        
        for i, doc in enumerate(docs_ref, 1):
            data = doc.to_dict()
            # Assuming your documents have 'question' and 'answer' fields. 
            # Adjust these keys if your data structure is different.
            question = data.get('question', 'N/A')
            answer = data.get('answer', 'N/A')
            
            knowledge_text += f"Document ID: {doc.id}\nQuestion: {question}\nAnswer: {answer}\n---\n"
            
        return knowledge_text.strip()
    except Exception as e:
        print(f"Error fetching data from Firestore collection '{COLLECTION_NAME}': {e}")
        return "Knowledge Base: Failed to load external data due to a database error."

# Load the knowledge base once when the application starts
KNOWLEDGE_BASE = load_chatbot_knowledge()
print(f"Loaded knowledge base (approx {len(KNOWLEDGE_BASE)} characters).")

# --- 2. GEMINI CONFIGURATION ---

app = Flask(__name__)

# NOTE: Using a hardcoded API key is a security risk. 
# Use environment variables for production applications.
api_key = "AIzaSyCXmEx12sKqnKeAUbljyGJjUCs4IBFxDG8"
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

# System prompt is now ready to receive the KNOWLEDGE_BASE
BASE_SYSTEM_PROMPT = """
You are a compassionate mental health assistant specialized in helping people overcome porn addiction. 
Provide emotional support, motivational advice, and practical strategies.
Always respond in a non-judgemental, encouraging tone.

Use the provided KNOWLEDGE BASE below to answer specific questions, especially those related to 
internal resources or common resolutions. If the answer is not in the knowledge base, 
rely on your general helpful and supportive persona.
"""
# --- 3. FLASK APP ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

# Define a basic retry decorator for stability (useful for API calls)
def retry_on_error(max_attempts=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"Attempt {attempt + 1} failed. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

@app.route("/chat", methods=["POST"])
@retry_on_error(max_attempts=3)
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please type something."})

    # 1. Construct the full prompt, including the system prompt and the Firebase knowledge.
    # This is how the chatbot "reads" your Firebase data.
    full_prompt_context = f"{BASE_SYSTEM_PROMPT}\n\n--- KNOWLEDGE BASE START ---\n{KNOWLEDGE_BASE}\n--- KNOWLEDGE BASE END ---\n"
    
    # 2. Add the user's message to the context
    final_prompt = f"{full_prompt_context}\n\nUser Query: {user_message}\nAssistant Response:"

    try:
        response = model.generate_content(final_prompt)
        bot_reply = response.text if response.text else "⚠️ No response received from the AI model."
    except Exception as e:
        bot_reply = f"⚠️ I encountered an error while processing your request. ({str(e)})"

    print(f"User: {user_message}")
    print(f"Bot: {bot_reply}")

    return jsonify({"response": bot_reply})


if __name__ == "__main__":
    app.run(debug=True)