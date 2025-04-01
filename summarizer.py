import time
import re
import langdetect
from flask import Flask, request
from transformers import pipeline

app = Flask(__name__)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_text(text, max_length=300, min_length=100):
    text = clean_text(text)

    try:
        langs = langdetect.detect_langs(text)
        lang = langs[0].lang if langs else "unknown"
        confidence = langs[0].prob if langs else 0

        if lang != "en" or confidence < 0.9:
            return f"\n⚠️ Error: This summarization model only supports English text.\nDetected: {lang} ({confidence:.2f})\n"

    except Exception as e:
        return f"\n⚠️ Error: Unable to detect language.\n{str(e)}\n"

    word_count = len(text.split())

    if word_count < min_length:
        return f"\n⚠️ Error: The text is too short for summarization.\nMinimum words required: {min_length}, but got {word_count}.\n"

    if word_count > 1024:
        return f"\n⚠️ Error: The text is too long (max 1024 words).\nCurrent word count: {word_count}.\n"

    min_length = max(10, word_count // 2)
    max_length = min_length * 2

    start_time = time.time()
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    end_time = time.time()

    time_taken = round(end_time - start_time, 2)

    output = (
        f"\nTime To Think: {time_taken} seconds\n"
        f"{'-' * 36}\n"
        f"Summary:\n"
        f"{summary[0]['summary_text']}\n"
    )

    return output

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return "\n⚠️ Please enter text\n"

    result = summarize_text(text)
    return result

if __name__ == '__main__':
    app.run(debug=True)
