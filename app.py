from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

api_key = "AIzaSyCXmEx12sKqnKeAUbljyGJjUCs4IBFxDG8"
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

# System prompt for mental health & porn addiction support
system_prompt = """
You are a compassionate mental health assistant specialized in helping people overcome porn addiction. 
Provide emotional support, motivational advice, and practical strategies.
Always respond in a non-judgmental, encouraging tone.
"""

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please type something."})

    try:
        # Combine system prompt with user message
        full_prompt = system_prompt + "\nUser: " + user_message + "\nAssistant:"

        response = model.generate_content(full_prompt)
        bot_reply = response.text if response.text else "⚠️ No response received."
    except Exception as e:
        bot_reply = f"⚠️ Error: {str(e)}"

    print(f"User: {user_message}")
    print(f"Bot: {bot_reply}")

    return jsonify({"response": bot_reply})


if __name__ == "__main__":
    app.run(debug=True)
