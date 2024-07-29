"""
This starts a Flask server that downloads a PDF from a given URL,
chunks the PDF into paragraphs,
vectorizes the paragraphs using a pre-trained Sentence Transformer model,
and stores the chunks and vectors in a PostgreSQL database.
"""
import json
import re
from flask import Flask, Response, request, jsonify
import requests
import pymupdf
from sentence_transformers import SentenceTransformer, util
from pgvector.psycopg2 import register_vector
import numpy as np
import psycopg2
app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define sample intents
intents = [
    "I have a PDF document",
    "Can you find information in this PDF file?",
    "Here is a link to a PDF",
    "Please look up this PDF"
]
intent_embeddings = model.encode(intents, convert_to_tensor=True)

# Database connection
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="eiw6MjoK32TUpauTxyZV",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Enable pgvector extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()
register_vector(conn)

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS pdf_chunks (
    id SERIAL PRIMARY KEY,
    chunk TEXT,
    vector VECTOR(384),
    page_number INT
)
""")
conn.commit()

def is_pdf(url):
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        for chunk in response.iter_content(chunk_size=1024):
            if b'%PDF' in chunk:
                return True
            break
    except requests.RequestException:
        return False
    return False

def detect_pdf_intent_advanced(messages):
    for message in messages:
        if message.get('role') == 'user':
            content = message.get('content', '')
            words = content.split()
            
            for word in words:
                if word.startswith("http"):
                    # Check for intent to reference a PDF
                    content_embedding = model.encode(content, convert_to_tensor=True)
                    cosine_scores = util.pytorch_cos_sim(content_embedding, intent_embeddings)
                    
                    if cosine_scores.max().item() > 0.5 and is_pdf(word):  # Threshold for intent detection
                        return word
    return None

def download_pdf(url):
    response = requests.get(url)
    with open('temp.pdf', 'wb') as f:
        f.write(response.content)
    return 'temp.pdf'

def chunk_pdf(file_path):
    doc = pymupdf.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        # Regex-based chunking
        regex_chunks = re.split(r'\n{2,}', text)
        for chunk in regex_chunks:
            chunks.append((chunk.strip(), page_num))

    return chunks

def vectorize_chunks(chunks):
    vectors = model.encode([chunk[0] for chunk in chunks])
    return [(chunks[i][0], vectors[i], chunks[i][1]) for i in range(len(chunks))]

def store_chunks(chunks):
    for chunk, vector, page_num in chunks:
        cursor.execute(
            """
            INSERT INTO pdf_chunks (chunk, vector, page_number)
            VALUES (%s, %s, %s)
            """,
            (chunk, vector.tolist(), page_num)
        )
    conn.commit()

def process_pdf_and_store_chunks(pdf_url):
    pdf_path = download_pdf(pdf_url)
    chunks = chunk_pdf(pdf_path)
    vectorized_chunks = vectorize_chunks(chunks)
    store_chunks(vectorized_chunks)

def perform_similarity_search(query_vector):
    cursor.execute("""
        SELECT chunk, page_number, vector <-> %s::vector AS distance
        FROM pdf_chunks
        ORDER BY distance
        LIMIT 1
    """, (query_vector,))
    result = cursor.fetchone()
    return result if result else (None, None)

def exclude_headers(headers):
    #region exlcude some keys in :res response
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']  #NOTE we here exclude all "hop-by-hop headers" defined by RFC 2616 section 13.5.1 ref. https://www.rfc-editor.org/rfc/rfc2616#section-13.5.1
    headers          = [
        (k,v) for k,v in headers.items()
        if k.lower() not in excluded_headers
    ]
    #endregion exlcude some keys in :res response
    return headers

@app.route('/api/chat', methods=['POST'])
def chat_completions():
    data = json.loads(request.data.decode('utf-8'))
    messages = data.get('messages', [])

    # Detect intent and find PDF link using the advanced method
    pdf_url = detect_pdf_intent_advanced(messages)

    # # Check the latest user message for a PDF link
    # pdf_link_pattern = r'(https?://\S+\.pdf)'
    # pdf_links = []
    # for message in messages:
    #     if message.get('role') == 'user':
    #         pdf_links.extend(re.findall(pdf_link_pattern, message.get('content', '')))

    if pdf_url:
        # Process the PDF and store chunks
        process_pdf_and_store_chunks(pdf_url)
        # Vectorize the user query (the latest user message) to perform similarity search
        user_message = messages[-1]['content']
        query_vector = model.encode([user_message])[0].tolist()
        chunk, page_number, score = perform_similarity_search(query_vector)

        if chunk:
            # Modify the user message to include the most similar chunk
            rag_content = f"Chunk: {chunk}\n\nUser Query: {user_message}"
            messages[-1]['content'] = rag_content

    # Update the payload with the modified messages
    data['messages'] = messages

    # Forward the request to the actual model server
    res = requests.post('http://localhost:11434/api/chat', json=data, timeout=60)

    response = Response(res.content, res.status_code, exclude_headers(res.raw.headers))
    return response


@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    url = f'http://localhost:11434/{path}'

    res = requests.request(  # ref. https://stackoverflow.com/a/36601467/248616
        method          = request.method,
        # url             = request.url.replace(request.host_url, f'{OLLAMA_URL}/'),
        url             = url,
        headers         = {k:v for k,v in request.headers if k.lower() != 'host'}, # exclude 'host' header
        data            = request.get_data(),
        cookies         = request.cookies,
        allow_redirects = False,
    )


    response = Response(res.content, res.status_code, exclude_headers(res.raw.headers))
    return response

    # Forward all other requests to the actual model server
   
    headers = {'Content-Type': 'application/json'}

    if request.method == 'GET':
        resp = requests.get(url, params=request.args, timeout=60, headers=headers)
    elif request.method == 'POST':
        # resp = requests.post(url, json=request.json, timeout=60, headers=headers)
        resp = requests.post(url, request=request)
    elif request.method == 'PUT':
        resp = requests.put(url, json=request.json, timeout=60, headers=headers)
    elif request.method == 'DELETE':
        resp = requests.delete(url, json=request.json, timeout=60, headers=headers)
    else:
        return jsonify({"error": "Unsupported HTTP method"}), 405

    return (resp.text, resp.status_code, resp.headers.items())

if __name__ == '__main__':
    app.run(debug=True)
