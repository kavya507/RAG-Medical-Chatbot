from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os
import traceback

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
app = Flask(__name__)
app.secret_key = os.urandom(24)

from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace("\n", "<br>"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/", methods = ["GET", "POST"])
def index():
    if "messages" not in session:
        session['messages'] = []
    
    if request.method == "POST":
        user_input = request.form.get("prompt")
        if user_input:
            messages = session['messages']
            messages.append({"role": "user", "content": user_input})
            session['messages'] = messages
            try: 
                qa_chain = create_qa_chain()
                response = qa_chain.invoke({'query': user_input})
                result = response.get("result", "No response")
                if not result:
                    result = "Sorry, I couldn’t find an answer to your question."
                messages.append({"role": "assistant", "content":result})
                session['messages'] = messages
            except Exception as e:
                error_message = f"Error: {str(e)}"
                if not str(e).strip():
                    error_message = f"Error: {type(e).__name__}"
                    error_message += f"\nTraceback:\n{traceback.format_exc()}"
                return render_template("index.html", messages=session["messages"], error=error_message)
        
        return redirect(url_for("index")) 
    return render_template("index.html", messages=session.get("messages", []))

@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5001, debug=False, use_reloader = False)
            
