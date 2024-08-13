#  Retrieval Augmented Generation Chatbot by CNASG - IPEX-LLM (CPU Only)
This project leverages the power of Intel’s IPEX-LLM to perform Retrieval Augmented Generation (RAG) efficiently on CPU-only systems. This demo provides a RAG chatbot that aims to generate contextually relevant text without the need for GPU acceleration.

## Start with Docker (the easy way)
Use the pre-built images prepared to quickly serve up the demo on your local machine.\
The container comes bundled with the respective Transformer LLM + Embedding models, and uses a in-memory vector database.
### Pull Docker Image
```bash
docker pull quay.io/yxchia98/ipex-llm-qwen2-rag:latest
```
### Run Docker Container
```bash
docker run --rm --net host quay.io/yxchia98/ipex-llm-qwen2-rag:latest
```
If you're in an environment where you cant expose port 7860, run the following command where you expose it on port 80:
```bash
docker run --rm -p 80:7860 quay.io/yxchia98/ipex-llm-qwen2-rag:latest
```
### Accessing the RAG Chatbot Demo
The chatbot will be started on localhost port 7860 (default gradio port)\
To access the chatbot, simply access: http://127.0.0.1:7860/

## The Manual Way
This process brings you through the following steps
1. Downloading the Transformer LLM (Qwen/Qwen2-1.5B-Instruct) & Embedding model (BAAI/bge-small-en-v1.5) from HuggingFace
2. Pull and start IPEX-LLM CPU container image `intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT`, mounting model & script folders as volumes
3. Install needed dependencies (llama-index & ipex-llm)
4. Convert the HF Transformer LLM model into IPEX-LLM models
5. Start RAG Chatbot

### Clone repository
```bash
git clone https://github.com/yxchia98/ipex-llm-demo-cnasg.git
```

### Downloading & preparing models
```bash
pip install huggingface_hub
python download_model.py
```

### Start IPEX-LLM container & install dependencies
```bash
docker run -d -it --net host -v ../demo:/demo -v ../llm-models:/llm-models intelanalytics/ipex-llm-cpu:ipex-llm-cpu:2.1.0-SNAPSHOT
```
```bash
docker exec -it <container-id> bash
```
```bash
pip install llama-index-llms-ipex-llm llama-index-embeddings-ipex-llm llama-index-readers-web llama-index gradio
pip install ipex-llm==2.1.0b20240712
pip install -U transformers==4.37.0 tokenizers==0.15.2
``` 

### Convert HF transformer model to IPEX-LLM
```bash
cd /demo
python convert-model.py --repo-id-or-model-path /llm-models/hf-models/Qwen2-1.5B-Instruct --save-path /llm-models/ipex-models/Qwen2-1.5B-Instruct
```

### Start RAG Chatbot
```bash
python rag-demo.py
```

### Accessing the RAG Chatbot Demo
The chatbot will be started on localhost port 7860 (default gradio port)\
To access the chatbot, simply access: http://127.0.0.1:7860/