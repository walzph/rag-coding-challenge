# PDF-Enabled Proxy Server for Ollama

This project is a Flask-based proxy server that sits in front of an Ollama server. It provides PDF processing and retrieval-augmented generation (RAG) capabilities, allowing enhanced interactions through the Ollama API or compatible frontends like [Open WebUI](https://github.com/open-webui/open-webui).

## Key Features

- **Proxy Server**: Acts as a middleware between clients and an Ollama server, forwarding all requests by default.
- **PDF Processing**: Detects PDF links in user messages and automatically downloads, processes, and vectorizes the document.
- **RAG Context**: When a PDF is processed, its contents are used to enrich the context of subsequent chat interactions, improving the quality of responses.

### Missing
- [ ] Remember that a PDF (or link) has already been processed
    - [ ] Add a content hash column to the table
    - [ ] Cache downloads, etc.
 
## Usage

- **Direct Access**: Interact with the proxy server directly via the Ollama API.
- **Frontends**: Integrate with interfaces such as Open WebUI for seamless interactions.

## How It Works

1. **Intent Detection**: The server analyzes user messages for intent to reference a PDF.
2. **PDF Download and Chunking**: If a PDF link is detected, the document is downloaded and chunked into paragraphs.
3. **Vectorization**: Chunks are vectorized using a pre-trained Sentence Transformer model and stored in a PostgreSQL database.
4. **RAG Integration**: Relevant PDF content is used as context in subsequent chat messages to enhance responses.

## Setup

1. **Install Dependencies**: Ensure Python, Flask, and PostgreSQL are installed.
2. **Database Configuration**: Set up a PostgreSQL database and update connection details in the code.
3. **Run the Server**: Start the Flask server using `python app.py`.

## Example

To interact with the server, send chat completion requests to the proxy endpoint. If a PDF link is included, the server processes the document and enhances the chat context accordingly.

