from flask import Flask, request, jsonify
import requests
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import json

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Database connection
conn = psycopg2.connect(
    dbname="your_dbname",
    user="your_username",
    password="your_password",
    host="your_host",
    port="your_port"
)
register_vector(conn)
cursor = conn.cursor()

# Enable pgvector extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS pdf_chunks (
    id SERIAL PRIMARY KEY,
    chunk TEXT,
    vector VECTOR(384),  # Adjust the dimension according to your vector size
    page_number INT
)
""")
conn.commit()

def download_pdf(url):
    response = requests.get(url)
    with open('temp.pdf', 'wb') as f:
        f.write(response.content)
    return 'temp.pdf'

def chunk_pdf(file_path):
    doc = fitz.Document(file_path)
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

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data.get('prompt', '')

    # Check if the prompt contains a PDF link
    pdf_link_pattern = r'(https?://\S+\.pdf)'
    pdf_links = re.findall(pdf_link_pattern, prompt)

    if pdf_links:
        pdf_url = pdf_links[0]
        # Process the PDF and store chunks
        process_pdf_and_store_chunks(pdf_url)
        # Vectorize the prompt to perform similarity search
        query_vector = model.encode([prompt])[0].tolist()
        most_similar_chunk, page_number = perform_similarity_search(query_vector)

        if most_similar_chunk:
            # Modify the prompt to include the most similar chunk
            prompt = f"Chunk: {most_similar_chunk}\n\nUser Query: {prompt}"
            data['prompt'] = prompt

    # Forward the request to the actual model server
    response = requests.post('http://llama_cpp_endpoint/v1/completions', json=data)
    return jsonify(response.json())

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    # Forward all other requests to the actual model server
    if request.method == 'GET':
        resp = requests.get(f'http://llama_cpp_endpoint/{path}', params=request.args)
    elif request.method == 'POST':
        resp = requests.post(f'http://llama_cpp_endpoint/{path}', json=request.json)
    elif request.method == 'PUT':
        resp = requests.put(f'http://llama_cpp_endpoint/{path}', json=request.json)
    elif request.method == 'DELETE':
        resp = requests.delete(f'http://llama_cpp_endpoint/{path}', json=request.json)
    else:
        return jsonify({"error": "Unsupported HTTP method"}), 405

    return (resp.text, resp.status_code, resp.headers.items())

if __name__ == '__main__':
    app.run(debug=True)
