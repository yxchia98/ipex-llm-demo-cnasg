# import
from llama_index.core import StorageContext, Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ipex_llm import IpexLLM
from llama_index.embeddings.ipex_llm import IpexLLMEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import time
import gradio as gr
from tqdm import tqdm
import shutil
import os
from pathlib import Path
import glob


os.environ['GRADIO_TEMP_DIR'] = "/tmp/gradio"

RAG_UPLOAD_FOLDER = "/demo/rag-documents/"




class Custom_Query_Engine():
    def __init__(self):
        self.SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner. Here are some rules you always follow:
        - Generate human readable output, avoid creating output with gibberish text.
        - Generate only the requested output, don't include any other language before or after the requested output.
        """
        # self.hf_model_path = "/llm-models/hf-models/Qwen2-1.5B-Instruct"
        self.saved_lowbit_model_path = "/llm-models/ipex-models/Qwen2-1.5B-Instruct"
        

        self.llm = IpexLLM.from_model_id_low_bit(
            model_name=self.saved_lowbit_model_path,
            # tokenizer_name=self.hf_model_path,
            tokenizer_name=self.saved_lowbit_model_path,  # copy the tokenizers to saved path if you want to use it this way
            context_window=4096,
            max_new_tokens=2048,
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            completion_to_prompt=self.completion_to_prompt,
            messages_to_prompt=self.messages_to_prompt,
            )

        self.embed_model = IpexLLMEmbedding(model_name="/llm-models/hf-models/bge-small-en-v1.5", trust_remote_code=True)
            
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # self.remote_db = chromadb.HttpClient(host=f"https://chromadb.cnasg.dellcsc.com:443")
        self.remote_db = chromadb.HttpClient(host=f"{os.environ['CHROMA_DB_URL']}:{os.environ['CHROMA_DB_PORT']}")

        # check and recreate collection if exists
        if "rag_data" in [c.name for c in self.remote_db.list_collections()]:
            print('collection rag_data exists, deleting...')
            self.remote_db.delete_collection(name="rag_data")
            print('deleted collection rag_data!')

        self.chroma_collection = self.remote_db.get_or_create_collection("rag_data")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)


        self.documents = SimpleDirectoryReader("./data/").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=self.storage_context, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=2)
        self.use_rag = False

    def toggle_rag(self, toggle):
        self.use_rag = toggle
        return self.use_rag

    def get_rag_toggle(self):
        return self.use_rag

    def reload(self, path):
        del self.query_engine
        del self.index
        self.documents = SimpleDirectoryReader(RAG_UPLOAD_FOLDER).load_data()
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=self.storage_context, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=2)

    def query(self, message):
        return self.query_engine.query(message)

    def query_without_rag(self, message):
        # self.query_engine = self.llm.as_query_engine(streaming=True)
        return self.llm.stream_complete(message)
        # return self.query_engine.query(message)

    def completion_to_prompt(self, completion):
        # print(f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n")
        return f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

    def messages_to_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
            elif message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
            elif message.role == "assistant":
                prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

        if not prompt.startswith("<|im_start|>system"):
            prompt = "<|im_start|>system\n" + prompt

        prompt = prompt + "<|im_start|>assistant\n"

        # print(prompt)

        return prompt
    

         


# SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
# - Generate human readable output, avoid creating output with gibberish text.
# - Generate only the requested output, don't include any other language before or after the requested output.
# - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
# - Generate professional language typically used in business documents in North America.
# - Never generate offensive or foul language.
# """


# load documents
# documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
# documents = SimpleWebPageReader(html_to_text=True).load_data(
#     ["http://paulgraham.com/worked.html", "https://jujutsu-kaisen.fandom.com/wiki/Satoru_Gojo"]
# )
# documents = BeautifulSoupWebReader().load_data(
#     ["http://paulgraham.com/worked.html", "https://jujutsu-kaisen.fandom.com/wiki/Satoru_Gojo"]
# )


css = """
.app-interface {
    height:90vh;
}
.chat-interface {
    height: 85vh;
}
.file-interface {
    height: 40vh;
}
.web-interface {
    height: 30vh;
}
"""

query_engine = Custom_Query_Engine()

def stream_response(message, history):
    print(f"current RAG toggle is {query_engine.get_rag_toggle()}")
    if query_engine.get_rag_toggle():
        print('using RAG')
        response = query_engine.query(message)
        res = ""
        for token in response.response_gen:
            # print(token, end="")
            res = str(res) + str(token)
            yield res
    else:
        print('not using RAG')
        response = query_engine.query_without_rag(message)
        res = ""
        for token in response:
            # print(token, end="")
            res = str(res) + str(token.delta)
            yield res

def vectorize(files, progress=gr.Progress()):
    Path(RAG_UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    UPLOAD_FOLDER = RAG_UPLOAD_FOLDER

    prev_files = glob.glob(f"{UPLOAD_FOLDER}*")
    for f in prev_files:
        os.remove(f)

    if not files:
        return []
    
    file_paths = [file.name for file in files]
    for file in files:
        shutil.copy(file.name, UPLOAD_FOLDER)

    query_engine.reload(UPLOAD_FOLDER)
    
    return file_paths

def toggle_knowledge_base(use_rag):
    print(f"toggling use knowledge base to {use_rag}")
    query_engine.toggle_rag(use_rag)
    return use_rag


with gr.Blocks(css=css) as demo:
    # gr.Markdown(
    # """
    # # **Retrieval Augmented Generation with only CPU**
    # """)
    gr.Markdown(
    """
    <h1 style="text-align: center;">Retrieval Augmented Generation with only CPU 💻📑✨</h3>
    """)
    with gr.Row(equal_height=True, elem_classes=["app-interface"]):
        with gr.Column(scale=4, elem_classes=["chat-interface"]):
            test = gr.ChatInterface(fn=stream_response)
        with gr.Column(scale=1):
            file_input = gr.File(elem_classes=["file-interface"], file_types=["pdf", "csv", "text", "html"], file_count="multiple")
            # upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video", "pdf", "csv", "text"], file_count="multiple")
            # upload_button.upload(upload_file, upload_button, file_input)
            vectorize_button = gr.Button("Vectorize Files")
            vectorize_button.click(fn=vectorize, inputs=file_input, outputs=file_input)
            use_rag = gr.Checkbox(label="Use Knowledge Base")
            use_rag.select(fn=toggle_knowledge_base, inputs=use_rag)
            

demo.launch(server_name="0.0.0.0", ssl_verify=False)



# what difference does dell technologies make in on-premise inferencing?


