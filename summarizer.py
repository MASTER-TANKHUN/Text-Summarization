import nltk
import re
from transformers import pipeline

# โหลดโมเดลสรุปข้อความ
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ฟังก์ชันล้างข้อความ
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # ลบช่องว่างเกิน
    return text.strip()

# ฟังก์ชันสรุปเนื้อหา
def summarize_text(text, max_length=150):
    text = clean_text(text)
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']