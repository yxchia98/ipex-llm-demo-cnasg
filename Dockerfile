FROM intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT

RUN pip install llama-index-llms-ipex-llm llama-index-embeddings-ipex-llm llama-index-readers-web llama-index gradio
RUN pip install ipex-llm==2.1.0b20240712
RUN pip install -U transformers==4.37.0 tokenizers==0.15.2

COPY demo/ /demo/
COPY llm-models/ /llm-models/ 

ENTRYPOINT [ "/bin/bash" ]
CMD ["-c", "cd /demo/ && python rag-demo.py"]