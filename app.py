from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from functools import wraps
import time
import re # Needed for simple answer parsing

# --- Firebase Initialization ---
try:
    # NOTE: In a real environment, you would use os.environ or Flask config for API keys/paths.
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}. Ensure 'firebase_key.json' is correct.")
    db = None 

def load_chatbot_knowledge():
    """
    Fetches the data from Firestore, formats it, and returns it as a string 
    to be used as context for the Gemini model.
    """
    if not db:
        return "Knowledge Base: Failed to load external data (Firebase initialization failed)."
    knowledge_text = "KNOWLEDGE BASE DATA FOR CHATBOT:\n"
    COLLECTION_NAME = 'support_documents' 
    try:
        docs_ref = db.collection(COLLECTION_NAME).stream()
        for i, doc in enumerate(docs_ref, 1):
            data = doc.to_dict()
            question = data.get('question', 'N/A')
            answer = data.get('answer', 'N/A')
            knowledge_text += f"Document ID: {doc.id}\nQuestion: {question}\nAnswer: {answer}\n---\n"
        return knowledge_text.strip()
    except Exception as e:
        print(f"Error fetching data from Firestore collection '{COLLECTION_NAME}': {e}")
        return "Knowledge Base: Failed to load external data due to a database error."

KNOWLEDGE_BASE = load_chatbot_knowledge()
print(f"Loaded knowledge base (approx {len(KNOWLEDGE_BASE)} characters).")

# --- App and Generative AI Setup ---
app = Flask(__name__)
# API key kept as provided by the user
api_key = "AIzaSyCXmEx12sKqnKeAUbljyGJjUCs4IBFxDG8" 
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

BASE_SYSTEM_PROMPT = """
You are a compassionate mental health assistant specialized in helping people overcome porn addiction. 
Provide emotional support, motivational advice, and practical strategies.
Always respond in a non-judgemental, encouraging tone.

Use the provided KNOWLEDGE BASE below to answer specific questions, especially those related to 
internal resources or common resolutions. If the answer is not in the knowledge base, 
rely on your general helpful and supportive persona.
"""

# --- Global State for Single-User PDI Assessment ---
PDI_STATE = {
    'active': False,
    'q_index': -1, # -1 means not started, 0 to 7 is active, 8 is finished (needs reset)
    'score': 0,
}


# --- PDI Assessment Data ---
PDI_QUESTIONS = [
    # Q1: Usage Patterns
    {"q": "In a typical week, how often do you watch pornography?", "section": "Usage Patterns",
     "options": {"A": 0, "B": 1, "C": 2, "D": 3},
     "text": [("A", "Rarely, or not every week"), ("B", "1-3 times a week"), ("C", "4-6 times a week"), ("D", "Daily, or multiple times a day")]},
    
    # Q2: Usage Patterns
    {"q": "When you do watch, how long does a typical session last?", "section": "Usage Patterns",
     "options": {"A": 0, "B": 1, "C": 2, "D": 3},
     "text": [("A", "Less than 15 minutes"), ("B", "15 to 45 minutes"), ("C", "About an hour"), ("D", "More than an hour")]},

    # Q3: Loss of Control & Compulsivity
    {"q": "Do you often find yourself watching porn for much longer than you originally intended?", "section": "Loss of Control & Compulsivity",
     "options": {"A": 0, "B": 1, "C": 2, "D": 3},
     "text": [("A", "Never"), ("B", "Sometimes"), ("C", "Often"), ("D", "Almost every time")]},

    # Q4: Loss of Control & Compulsivity
    {"q": "Have you tried to stop or cut down on watching, but found you couldn't?", "section": "Loss of Control & Compulsivity",
     "options": {"A": 1, "B": 2, "C": 3, "D": 4},
     "text": [("A", "I have never tried to stop."), ("B", "I've tried and it was manageable."), ("C", "I've tried and it was very difficult."), ("D", "I've tried multiple times and failed.")]},

    # Q5: Psychological Reliance & Triggers
    {"q": "What is the most common reason you turn to pornography?", "section": "Psychological Reliance & Triggers",
     "options": {"A": 0, "B": 1, "C": 2, "D": 3},
     "text": [("A", "Sexual curiosity or entertainment"), ("B", "Habit or boredom"), ("C", "To cope with stress or anxiety"), ("D", "To escape feelings of sadness, loneliness, or anger")]},

    # Q6: Psychological Reliance & Triggers
    {"q": "After watching, how do you typically feel about yourself?", "section": "Psychological Reliance & Triggers",
     "options": {"A": 0, "B": 1, "C": 2, "D": 3},
     "text": [("A", "Fine, or positive"), ("B", "Indifferent or empty"), ("C", "A little guilty or regretful"), ("D", "Overwhelmed with shame, anxiety, or disgust")]},

    # Q7: Negative Consequences
    {"q": "Has your pornography use negatively affected your real-life relationships, work, or studies?", "section": "Negative Consequences",
     "options": {"A": 0, "B": 1, "C": 2, "D": 3},
     "text": [("A", "No, I don't believe so."), ("B", "It has caused minor issues or arguments."), ("C", "It has caused significant problems (e.g., loss of focus, hiding the behavior)."), ("D", "It has directly damaged a relationship, my job, or my academic performance.")]},

    # Q8: Negative Consequences
    {"q": "Do you find yourself thinking about pornography when you should be focusing on other things (like work, conversations, or hobbies)?", "section": "Negative Consequences",
     "options": {"A": 0, "B": 1, "C": 2, "D": 3},
     "text": [("A", "Rarely or never"), ("B", "Sometimes"), ("C", "Often, it's distracting"), ("D", "Constantly, it's difficult to think about anything else")]}
]

def get_pdi_analysis(score):
    """Calculates PDI level, interpretation, and app action based on the score."""
    if 0 <= score <= 6:
        level = "Low Dependability"
        interpretation = "Your usage appears to be controlled and is likely not causing significant issues in your life. You may be here for curiosity or to build healthier habits."
        action = "The app can recommend foundational content on mindful internet use, goal setting, and channel-switching techniques. The approach can be less intensive."
    elif 7 <= score <= 13:
        level = "Moderate Dependability"
        interpretation = "Your habit is becoming more established. You may be feeling a loss of control and experiencing some negative consequences. This is a crucial stage to build awareness and new coping mechanisms."
        action = "The app should suggest a structured program, introduce CBT exercises for identifying triggers, and strongly encourage using the 'Urge Log' and 'Community' features."
    elif 14 <= score <= 20:
        level = "High Dependability"
        interpretation = "Your pornography use is likely a primary coping mechanism and is having a clear, negative impact on your life. The behavior may feel compulsive and difficult to manage on your own."
        action = "The app should immediately recommend a more intensive, structured daily plan. It should prioritize features like accountability partners, emergency 'parachute' options, and advanced content on neuroscience and recovery. It could also provide resources for finding a therapist."
    else: # 21-26
        level = "Severe Dependability"
        interpretation = "Your relationship with pornography is causing significant distress and disruption. The behavior is likely compulsive, and you may feel powerless to stop. Professional help is strongly recommended."
        action = "The app should present its most robust features immediately. The tone should be highly supportive but firm. Most importantly, it should prominently display resources for professional help, such as links to therapists specializing in addiction, support groups (like SAA), and mental health hotlines. The app serves as a powerful tool, but it should encourage professional guidance at this level."
    
    # Format the final analysis nicely
    analysis_text = f"### 1. Your Assessment Results\n\n"
    analysis_text += f"**Total Score:** {score} points (out of 26)\n"
    analysis_text += f"**PDI Level:** {level}\n\n"
    analysis_text += f"### 2. Interpretation of Your PDI Level\n\n"
    analysis_text += f"_{interpretation}_\n\n"
    analysis_text += f"### 3. Recommended Next Steps (App Action)\n\n"
    analysis_text += f"Here is the recommended action plan for your current level:\n"
    # Convert action string to a bulleted list based on sentences for better display
    actions_list = action.split('. ')
    actions_bulleted = "\n".join([f"* {item.strip()}" for item in actions_list if item.strip()])
    analysis_text += actions_bulleted
    
    return analysis_text


def pdi_ask_next_question():
    """Returns the text for the current/next PDI question."""
    idx = PDI_STATE['q_index']
    if idx >= len(PDI_QUESTIONS):
        return None # Should not happen if logic is correct
    
    q_data = PDI_QUESTIONS[idx]
    
    q_text = f"**Section {idx // 2 + 1}: {q_data['section']}**\n"
    q_text += f"**Question {idx + 1} of {len(PDI_QUESTIONS)}:** \"{q_data['q']}\"\n\n"
    
    for label, text in q_data['text']:
        points = q_data['options'][label]
        q_text += f"({label}) {text} ({points} points)\n"
        
    q_text += "\nPlease respond with the letter (A, B, C, or D) that best reflects your answer."
    return q_text

def pdi_process_answer(answer_text):
    """Processes the user's answer, updates score, and increments index."""
    idx = PDI_STATE['q_index']
    if idx < 0 or idx >= len(PDI_QUESTIONS):
        return None, "Error: Assessment state invalid."

    # Simple parsing to get the first valid letter A, B, C, or D, case-insensitive
    match = re.search(r'[a-d]', answer_text.upper())
    
    if not match:
        # Invalid answer, ask the current question again without changing state
        return pdi_ask_next_question(), "I didn't quite catch that. Please respond with the letter A, B, C, or D."

    user_choice = match.group(0)
    q_data = PDI_QUESTIONS[idx]
    
    if user_choice not in q_data['options']:
        # This should theoretically be covered by the regex, but as a safeguard
        return pdi_ask_next_question(), "That choice is not valid for this question. Please try again."

    # Update score and state
    points = q_data['options'][user_choice]
    PDI_STATE['score'] += points
    PDI_STATE['q_index'] += 1
    
    # Check if assessment is complete
    if PDI_STATE['q_index'] == len(PDI_QUESTIONS):
        PDI_STATE['active'] = False
        final_score = PDI_STATE['score']
        PDI_STATE['q_index'] = -1
        PDI_STATE['score'] = 0 # Reset state for next time
        return get_pdi_analysis(final_score), "Assessment Complete."
    
    # Otherwise, ask the next question
    return pdi_ask_next_question(), "Answer Accepted."


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    
    if not user_message:
        return jsonify({"response": "Please type something or select an option."})

    # --- Assessment Management Flow ---

    # 1. Start Assessment Command
    if user_message == "START_PDI_ASSESSMENT":
        PDI_STATE['active'] = True
        PDI_STATE['q_index'] = 0
        PDI_STATE['score'] = 0
        
        # Start with the first question
        response_text = pdi_ask_next_question()
        return jsonify({"response": response_text})

    # 2. In-Progress Assessment Answer
    if PDI_STATE['active']:
        next_q_text, status_message = pdi_process_answer(user_message)
        
        if "Assessment Complete" in status_message:
            # End of assessment, show analysis and the initial menu again
            return jsonify({"response": next_q_text, "show_menu": True})
        
        if "Answer Accepted" in status_message or "Error" in status_message:
             # Regular question progression
            return jsonify({"response": next_q_text})

    # --- Main Menu Options Handled by Gemini ---

    if user_message == "FEELING_URGES":
        # Specific prompt to guide the model's response for a strong urge
        llm_prompt = "The user is currently feeling strong urges and needs immediate coping strategies and high motivation. Respond immediately with 2-3 actionable steps and a powerful, non-judgemental message of support."
        
    elif user_message == "SHARE_PROGRESS":
        # Specific prompt to guide the model's response for sharing progress
        llm_prompt = "The user wants to share their thoughts or progress. Respond with an encouraging and open-ended question to help them reflect and feel heard, such as 'That's fantastic. What are you most proud of in the last day or week?'"
        
    else:
        # Regular chatbot flow
        llm_prompt = f"User Query: {user_message}\nAssistant Response:"


    # --- Generic LLM Response Flow ---
    full_prompt_context = f"{BASE_SYSTEM_PROMPT}\n\n--- KNOWLEDGE BASE START ---\n{KNOWLEDGE_BASE}\n--- KNOWLEDGE BASE END ---\n"
    final_prompt = f"{full_prompt_context}\n\n{llm_prompt}"

    try:
        # Simple retry logic for the API call 
        max_attempts = 3
        bot_reply = "⚠️ No response received from the AI model."
        for attempt in range(max_attempts):
            try:
                response = model.generate_content(final_prompt)
                bot_reply = response.text if response.text else "⚠️ No response received from the AI model."
                break # Success
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for LLM call: {e}")
                time.sleep(1 + attempt * 2) # Exponential backoff
        
    except Exception as e:
        bot_reply = f"I encountered a severe error while processing your request. ({str(e)})"

    return jsonify({"response": bot_reply})


if __name__ == "__main__":
    app.run(debug=True)
