# import
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ipex_llm import IpexLLM
from llama_index.embeddings.ipex_llm import IpexLLMEmbedding
import chromadb

def completion_to_prompt(completion):
    return f"<|system|>\n<|end|>\n<|user|>\n{completion}<|end|>\n<|assistant|>\n"


# Transform a list of chat messages into zephyr-specific input
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}<|end|>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}<|end|>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n<|end|>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "<|system|>\n" + SYSTEM_PROMPT + "\n<|user|>\n{query_str}\n<|assistant|>"
)

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

embed_model = IpexLLMEmbedding(model_name="/home/yxchia/llm-models/hf-models/bge-large-en-v1.5", trust_remote_code=True)

Settings.llm = llm
Settings.embed_model = embed_model

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

print(response)