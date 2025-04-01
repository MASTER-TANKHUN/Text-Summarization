from flask import Flask, request, jsonify, render_template
from summarizer import summarize_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        summary = summarize_text(text)
        return render_template("index.html", summary=summary, text=text)
    return render_template("index.html", summary=None)

if __name__ == "__main__":
    app.run(debug=True)