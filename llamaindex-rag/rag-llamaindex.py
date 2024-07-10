# import
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ipex_llm import IpexLLM
from llama_index.embeddings.ipex_llm import IpexLLMEmbedding
from llama_index.readers.web import SimpleWebPageReader
import chromadb


SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

# Transform a string into input zephyr-specific input
def completion_to_prompt(completion):
    res = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{completion}<|end|>\n<|assistant|>\n"
    print(res)
    return res


def messages_to_prompt(messages):
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
            "<|system|>\n{SYSTEM_PROMPT}<|end|>\n" + prompt
        )
    
    print(prompt)

    return prompt


hf_model_path = "/llm-models/hf-models/Phi-3-mini-4k-instruct"
saved_lowbit_model_path = "/llm-models/ipex-models/Phi-3-mini-4k-instruct"

llm = IpexLLM.from_model_id_low_bit(
    model_name=saved_lowbit_model_path,
    tokenizer_name=hf_model_path,
    # tokenizer_name=saved_lowbit_model_path,  # copy the tokenizers to saved path if you want to use it this way
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    completion_to_prompt=completion_to_prompt,
    messages_to_prompt=messages_to_prompt,

)

embed_model = IpexLLMEmbedding(model_name="/llm-models/hf-models/bge-large-en-v1.5", trust_remote_code=True)

Settings.llm = llm
Settings.embed_model = embed_model

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("For the author, what happened at interleaf?")
for token in response.response_gen:
    print(token, end="")