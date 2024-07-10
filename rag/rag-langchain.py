import warnings

from langchain.chains import LLMChain
from langchain_community.llms import IpexLLM
from langchain_core.prompts import PromptTemplate

import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import IpexLLMBgeEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser





# 1. Load, chunk and index the contents of the blog to create a retriever.
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
loader = WebBaseLoader("https://jujutsu-kaisen.fandom.com/wiki/Satoru_Gojo")
docs = loader.load()

# IpexLLM Embedding model
embedding_model = IpexLLMBgeEmbeddings(
    model_name="/home/yxchia/llm-models/hf-models/gte-large-en-v1.5",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True,},
)

# sentence = "IPEX-LLM is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency."
# query = "What is IPEX-LLM?"

# text_embeddings = embedding_model.embed_documents([sentence, query])
# print(f"text_embeddings[0][:10]: {text_embeddings[0][:10]}")
# print(f"text_embeddings[1][:10]: {text_embeddings[1][:10]}")

# query_embedding = embedding_model.embed_query(query)
# print(f"query_embedding[:10]: {query_embedding[:10]}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={'k': 4, 'fetch_k': 20})



warnings.filterwarnings("ignore", category=UserWarning, message=".*padding_mask.*")

# <|system|>
# You are a helpful assistant.<|end|>
# <|user|>
# Question?<|end|>
# <|assistant|>

template = """<|user|>
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer: <|end|>
<|assistant|>:"""
# template = """<|system|>
# You are an assistant for question-answering tasks.<|end|>
# <|user|>You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question}
# Context: {context}<|end|>
# <|assistant|>:"""
# template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assisstant\n:"
# template = "<bos><start_of_turn>user\nYou are a helpful assistant.\n{question}<end_of_turn>\n<start_of_turn>model\n:"

# prompt = PromptTemplate(template=template, input_variables=["question", "context"])
prompt = ChatPromptTemplate.from_template(template)


# # Load HF model
# llm = IpexLLM.from_model_id(
#     model_id="/home/yxchia/llm-models/hf-models/Phi-3-mini-4k-instruct,
#     model_kwargs={"temperature": 0, "max_length": 512, "trust_remote_code": True},
# )

# Convert and save model in low_bit format (used by IPEX-LLM)
saved_lowbit_model_path = "/home/yxchia/llm-models/ipex-models/Phi-3-mini-4k-instruct"
# llm.model.save_low_bit(saved_lowbit_model_path)

# Load low_bit model
llm = IpexLLM.from_model_id_low_bit(
    model_id=saved_lowbit_model_path,
    tokenizer_id="/home/yxchia/llm-models/hf-models/Phi-3-mini-4k-instruct",
    # tokenizer_name=saved_lowbit_model_path,  # copy the tokenizers to saved path if you want to use it this way
    model_kwargs={"temperature": 0, "max_length": 1024, "trust_remote_code": True},
)

# llm_chain = prompt | llm

# question = "What is dell technologies? can you explain what are some of the things they do?"
# output = llm_chain.invoke(question)


# example_messages = prompt.invoke(
#     {"context": "filler context", "question": "filler question"}
# )



def format_docs(docs):
    # eg = "\n\n".join(doc.page_content for doc in docs)
    # print(eg)
    # print("----------END OF CONTEXT----------")
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# rag_chain.invoke("What is Task Decomposition?")
rag_chain.invoke("Who is gojo?")