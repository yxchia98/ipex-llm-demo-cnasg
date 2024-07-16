# import
from llama_index.core import StorageContext, Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ipex_llm import IpexLLM
from llama_index.embeddings.ipex_llm import IpexLLMEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.web import BeautifulSoupWebReader


class Custom_Query_Engine():
    def __init__(self):
        self.SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner. Here are some rules you always follow:
        - Generate human readable output, avoid creating output with gibberish text.
        - Generate only the requested output, don't include any other language before or after the requested output.
        """
        self.hf_model_path = "/llm-models/hf-models/Phi-3-mini-4k-instruct"
        self.saved_lowbit_model_path = "/llm-models/ipex-models/Phi-3-mini-4k-instruct"
        

        self.llm = IpexLLM.from_model_id_low_bit(
            model_name=self.saved_lowbit_model_path,
            tokenizer_name=self.hf_model_path,
            # tokenizer_name=saved_lowbit_model_path,  # copy the tokenizers to saved path if you want to use it this way
            context_window=4096,
            max_new_tokens=2048,
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            completion_to_prompt=self.completion_to_prompt,
            messages_to_prompt=self.messages_to_prompt,
            )

        self.embed_model = IpexLLMEmbedding(model_name="/llm-models/hf-models/bge-small-en-v1.5", trust_remote_code=True)
            
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=2)

    def reload(self, path):
        del self.query_engine
        del self.index
        self.documents = SimpleDirectoryReader("/gradio/rag/").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=2)

    def query(self, message):
        return self.query_engine.query(message)

    def completion_to_prompt(self, completion):
        res = f"<|system|>\n{self.SYSTEM_PROMPT}<|end|>\n<|user|>\n{completion}<|end|>\n<|assistant|>\n"
        print(f"------------COMPLETION-TO-PROMPT------------------\n{res}")
        return res

    def messages_to_prompt(self, messages):
        prompt = ""
        system_found = False
        for message in messages:
            if message.role == "system":
                prompt += f"<|system|>\n{message.content}<|end|>\n"
                system_found = True
            elif message.role == "user":
                prompt += f"<|user|>\n{message.content}<|end|>\n"
            elif message.role == "assistant":
                prompt += f"<|assistant|>\n{message.content}<|end|>\n"
            else:
                prompt += f"<|user|>\n{message.content}<|end|>\n"

        # trailing prompt
        prompt += "<|assistant|>\n"

        if not system_found:
            prompt = (
                "<|system|>\nYou are a helpful AI assistant.<|end|>\n" + prompt
            )

        print(f"------------MESSAGES-TO-PROMPT------------------\n{prompt}")
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


import time
import gradio as gr
from tqdm import tqdm
import shutil
import os
from pathlib import Path
import glob


css = """
.app-interface {
    height:90vh;
}
.chat-interface {
    height: 90vh;
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
    response = query_engine.query(message)
    res = ""
    for token in response.response_gen:
        # print(token, end="")
        res = str(res) + str(token)
        yield res

def vectorize(files, progress=gr.Progress()):
    Path("/gradio/rag").mkdir(parents=True, exist_ok=True)
    UPLOAD_FOLDER = "/gradio/rag"

    prev_files = glob.glob(f"{UPLOAD_FOLDER}/*")
    for f in prev_files:
        os.remove(f)

    if not files:
        return []
    
    file_paths = [file.name for file in files]
    # for file in progress.tqdm(files, desc="Vectorizing..."):
    #     print(file.name, file)
    for file in files:
        shutil.copy(file.name, UPLOAD_FOLDER)

    # documents = SimpleDirectoryReader("/gradio/rag/").load_data()
    # index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
    query_engine.reload(UPLOAD_FOLDER)
    
    return file_paths


with gr.Blocks(css=css) as demo:
    with gr.Row(equal_height=True, elem_classes=["app-interface"]):
        with gr.Column(scale=4, elem_classes=["chat-interface"]):
            test = gr.ChatInterface(stream_response)
        with gr.Column(scale=1):
            file_input = gr.File(elem_classes=["file-interface"], file_types=["pdf", "csv", "text", "html"], file_count="multiple")
            # upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video", "pdf", "csv", "text"], file_count="multiple")
            # upload_button.upload(upload_file, upload_button, file_input)
            vectorize_button = gr.Button("Vectorize Files")
            vectorize_button.click(fn=vectorize, inputs=file_input, outputs=file_input)
            

demo.launch()



# what difference does dell technologies make in on-premise inferencing?


