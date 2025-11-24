import os
import gradio as gr
import re

print("🚀 Starting Text Summarizer...")

def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()

def extractive_summarize(text, sentences_count=3):
    """Simple extractive summarization using text rank algorithm"""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) <= sentences_count:
        return text
    
    # Simple scoring: sentence length + keyword frequency
    words = text.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Ignore short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    scored_sentences = []
    for sentence in sentences:
        score = len(sentence.split())  # Length score
        # Keyword score
        for word in sentence.lower().split():
            if word in word_freq and len(word) > 3:
                score += word_freq[word]
        scored_sentences.append((score, sentence))
    
    # Get top sentences (maintain order)
    scored_sentences.sort(reverse=True)
    top_sentences = scored_sentences[:sentences_count]
    top_sentences.sort(key=lambda x: sentences.index(x[1]))  # Restore original order
    
    summary = '. '.join([s[1] for s in top_sentences]) + '.'
    return summary

LENGTH_MAP = {
    "Short (2-3 sentences)": 2,
    "Medium (3-4 sentences)": 3, 
    "Long (4-5 sentences)": 4
}

def summarize_text(text, length):
    if not text:
        return "❌ Please paste some text."

    text = clean_text(text)
    
    try:
        if len(text.split()) < 50:
            return "📝 Text is too short. Please provide at least 50 words for better summarization."
        
        summary = extractive_summarize(text, LENGTH_MAP[length])
        return summary
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(title="📄 Text Summarizer") as app:
    gr.Markdown("## 📄 Text Summarizer")
    gr.Markdown("Simple extractive summarization - works instantly!")
    
    gr.Markdown("""
    **How it works:**
    - Extracts the most important sentences based on length and keyword frequency
    - No external APIs or model downloads
    - Works completely offline
    """)

    input_box = gr.Textbox(lines=12, label="Paste text", placeholder="Paste your article or long text here...")
    length_dd = gr.Dropdown(
        ["Short (2-3 sentences)", "Medium (3-4 sentences)", "Long (4-5 sentences)"],
        value="Medium (3-4 sentences)",
        label="Summary Length"
    )
    output_box = gr.Textbox(lines=10, label="Summary")

    summarize_btn = gr.Button("Summarize")
    summarize_btn.click(summarize_text, [input_box, length_dd], output_box)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        share=False
    )
