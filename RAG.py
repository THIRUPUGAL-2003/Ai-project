# Install required packages if not already installed:
# pip install flask ollama sentence-transformers faiss-cpu pypdf2

from flask import Flask, render_template, request, jsonify
import os
import mimetypes
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for RAG components
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Local embedding model
index = None
chunks = []  # Store text chunks
history = []  # Store question-answer history in memory (limited to 20 items)
is_pdf_uploaded = False  # Track PDF upload status
MAX_HISTORY = 20  # Limit history size

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            return "No extractable text found in the PDF."
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def chunk_text(text, chunk_size=512, overlap=128):
    """
    Improved chunking with overlap for better context retention.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks

def build_vector_store(text_chunks):
    global index, chunks
    if not text_chunks:
        logging.warning("No text chunks to build vector store.")
        return
    chunks = text_chunks
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    logging.info(f"Vector store built with {len(chunks)} chunks.")

@app.route('/')
def index_page():
    global history, is_pdf_uploaded
    # HTML template embedded as a string (optimized: write only if not exists)
    template_path = 'templates/index.html'
    os.makedirs('templates', exist_ok=True)
    if not os.path.exists(template_path):
        index_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG with Llama3 and Flask</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    margin: 0;
                    background-color: #1a1a1a;
                    color: #e0e0e0;
                    display: flex;
                    height: 100vh;
                    overflow: hidden;
                }
                .container {
                    display: flex;
                    width: 100%;
                }
                .sidebar {
                    width: 250px;
                    background-color: #222222;
                    padding: 20px;
                    overflow-y: auto;
                    border-right: 1px solid #333;
                }
                .sidebar h2 {
                    font-size: 1.2em;
                    margin-bottom: 20px;
                    color: #ffffff;
                }
                .history-item {
                    background-color: #333;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 0.9em;
                    word-wrap: break-word;
                }
                .history-item:hover {
                    background-color: #444;
                }
                .main-content {
                    flex-grow: 1;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }
                .answer-area {
                    flex-grow: 1;
                    padding: 20px;
                    overflow-y: auto;
                }
                .answer-area h2 {
                    font-size: 1.3em;
                    margin-bottom: 20px;
                    color: #ffffff;
                }
                #response {
                    background-color: #252525;
                    padding: 15px;
                    border-radius: 5px;
                    min-height: 100px;
                }
                .error {
                    color: #ff5555;
                }
                .info {
                    color: #55ff55;
                }
                .input-area {
                    padding: 20px;
                    background-color: #222222;
                    border-top: 1px solid #333;
                    display: flex;
                    align-items: center;
                }
                .input-container {
                    flex-grow: 1;
                    display: flex;
                    align-items: center;
                    background-color: #333;
                    border-radius: 25px;
                    padding: 10px;
                }
                textarea#question {
                    flex-grow: 1;
                    background: none;
                    border: none;
                    color: #e0e0e0;
                    font-size: 1em;
                    resize: none;
                    outline: none;
                    padding: 10px;
                    max-height: 100px;
                }
                #pdf {
                    display: none;
                }
                .icon-button {
                    background: none;
                    border: none;
                    cursor: pointer;
                    padding: 5px;
                    color: #e0e0e0;
                }
                .icon-button:hover {
                    color: #ffffff;
                }
                .file-label {
                    display: flex;
                    align-items: center;
                    cursor: pointer;
                }
                .file-label::before {
                    content: '📎';
                    font-size: 1.2em;
                    margin-right: 10px;
                }
                .send-button::before {
                    content: '➤';
                    font-size: 1.2em;
                }
                .clear-button::before {
                    content: '🗑️';
                    font-size: 1.2em;
                    margin-left: 10px;
                }
                .hidden {
                    display: none;
                }
                .loading::after {
                    content: ' ⏳';
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="sidebar">
                    <h2>History</h2>
                    <div id="history">
                        {% for item in history %}
                        <div class="history-item" data-answer="{{ item.answer | escape }}">{{ item.question | truncate(50) | escape }}</div>
                        {% endfor %}
                    </div>
                </div>
                <div class="main-content">
                    <div class="answer-area">
                        <h2>Responses</h2>
                        <div id="response"></div>
                    </div>
                    <div class="input-area">
                        <div class="input-container">
                            <label for="pdf" class="file-label icon-button {% if is_pdf_uploaded %}hidden{% endif %}" id="file-label"></label>
                            <input type="file" id="pdf" name="pdf" accept=".pdf">
                            <textarea id="question" name="question" placeholder="Ask a question..."></textarea>
                            <button class="icon-button clear-button" id="clear-button" title="Clear history"></button>
                            <button class="icon-button send-button" id="send-button"></button>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                // Toggle file input visibility
                function toggleFileInput() {
                    const fileLabel = document.getElementById('file-label');
                    fileLabel.classList.toggle('hidden', {{ is_pdf_uploaded | tojson }});
                }

                // Update history display
                function updateHistory(question, answer) {
                    if (question && answer) {
                        const historyDiv = document.getElementById('history');
                        const div = document.createElement('div');
                        div.className = 'history-item';
                        div.textContent = question.length > 50 ? question.substring(0, 47) + '...' : question;
                        div.dataset.answer = answer;
                        div.addEventListener('click', () => {
                            document.getElementById('response').innerHTML = `<p>${div.dataset.answer.replace(/\\n/g, '<br>')}</p>`;
                        });
                        historyDiv.appendChild(div);
                    }
                }

                // Handle PDF upload
                document.getElementById('pdf').addEventListener('change', async (e) => {
                    const formData = new FormData();
                    formData.append('pdf', e.target.files[0]);
                    const responseDiv = document.getElementById('response');
                    responseDiv.innerHTML = '<p class="loading">Uploading...</p>';
                    try {
                        const response = await fetch('/upload', {
                            method: 'POST',
                            body: formData
                        });
                        const result = await response.json();
                        responseDiv.innerHTML = response.ok
                            ? `<p class="info">${result.message}</p>`
                            : `<p class="error">${result.error}</p>`;
                        if (response.ok) {
                            toggleFileInput();
                        }
                    } catch (error) {
                        responseDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                    }
                });

                // Handle question submission
                document.getElementById('send-button').addEventListener('click', async () => {
                    const question = document.getElementById('question').value.trim();
                    if (!question) return;
                    const formData = new FormData();
                    formData.append('question', question);
                    const responseDiv = document.getElementById('response');
                    responseDiv.innerHTML = '<p class="loading">Processing...</p>';
                    try {
                        const response = await fetch('/ask', {
                            method: 'POST',
                            body: formData
                        });
                        const result = await response.json();
                        if (response.ok) {
                            responseDiv.innerHTML = `<p>${result.answer.replace(/\\n/g, '<br>')}</p>`;
                            updateHistory(question, result.answer);
                            document.getElementById('question').value = '';
                        } else {
                            responseDiv.innerHTML = `<p class="error">${result.error}</p>`;
                        }
                    } catch (error) {
                        responseDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                    }
                });

                // Handle clear history
                document.getElementById('clear-button').addEventListener('click', async () => {
                    try {
                        const response = await fetch('/clear_history', {
                            method: 'POST'
                        });
                        const result = await response.json();
                        if (response.ok) {
                            document.getElementById('history').innerHTML = '';
                            document.getElementById('response').innerHTML = `<p class="info">${result.message}</p>`;
                            toggleFileInput();
                        }
                    } catch (error) {
                        document.getElementById('response').innerHTML = `<p class="error">Error: ${error.message}</p>`;
                    }
                });

                // Initialize history click handlers
                document.querySelectorAll('.history-item').forEach(item => {
                    item.addEventListener('click', () => {
                        document.getElementById('response').innerHTML = `<p>${item.dataset.answer.replace(/\\n/g, '<br>')}</p>`;
                    });
                });

                // Initialize file input visibility
                toggleFileInput();
            </script>
        </body>
        </html>
        """
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(index_html)
    
    return render_template('index.html', history=history, is_pdf_uploaded=is_pdf_uploaded)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global is_pdf_uploaded
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Validate file type
    if not pdf_file.filename.lower().endswith('.pdf') or mimetypes.guess_type(pdf_file.filename)[0] != 'application/pdf':
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400
    
    pdf_path = os.path.join('uploads', pdf_file.filename)
    os.makedirs('uploads', exist_ok=True)
    pdf_file.save(pdf_path)
    
    text = extract_text_from_pdf(pdf_path)
    if text.startswith("Error") or text.startswith("No extractable text"):
        os.remove(pdf_path)  # Clean up invalid file
        return jsonify({'error': text}), 400
    
    text_chunks = chunk_text(text)
    build_vector_store(text_chunks)
    is_pdf_uploaded = True
    logging.info(f"PDF uploaded: {pdf_file.filename}")
    
    return jsonify({'message': 'PDF uploaded and processed successfully'})

@app.route('/ask', methods=['POST'])
def ask_question():
    global history
    question = request.form.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        if index is not None and chunks:
            # RAG mode: Retrieve context and generate answer
            question_embedding = embedder.encode([question])
            D, I = index.search(np.array(question_embedding).astype('float32'), k=3)
            relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            context = "\n\n".join(relevant_chunks)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question based only on the provided context. If the context doesn't have the information, say 'I don't have enough information from the document.'"
            logging.info("Using RAG mode for question.")
        else:
            # Non-RAG mode: Direct answer without context
            prompt = f"Question: {question}\n\nProvide a helpful and accurate answer."
            logging.info("Using direct mode (no PDF) for question.")
        
        response = ollama.generate(model='llama3', prompt=prompt)
        answer = response['response'].strip()
        
        # Store in history with limit
        history.append({'question': question, 'answer': answer})
        if len(history) > MAX_HISTORY:
            history.pop(0)
        
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return jsonify({'error': f"Error generating answer: {str(e)}"}), 500
    
    return jsonify({'answer': answer})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history, is_pdf_uploaded, index, chunks
    history = []
    is_pdf_uploaded = False
    index = None
    chunks = []
    # Optional: Clean up uploads folder
    if os.path.exists('uploads'):
        for file in os.listdir('uploads'):
            os.remove(os.path.join('uploads', file))
        os.rmdir('uploads')
    logging.info("History and state cleared.")
    return jsonify({'message': 'History and state cleared successfully'})

if __name__ == '__main__':
    app.run(debug=True)