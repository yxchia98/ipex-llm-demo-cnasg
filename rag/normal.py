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

template = """<|user|>
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm_chain = prompt | llm

question = "who is gojo?"
output = llm_chain.invoke(question)