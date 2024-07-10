from llama_index.embeddings.ipex_llm import IpexLLMEmbedding

embedding_model = IpexLLMEmbedding(model_name="/home/yxchia/llm-models/hf-models/bge-large-en-v1.5", trust_remote_code=True)

sentence = "IPEX-LLM is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency."
query = "What is IPEX-LLM?"

text_embedding = embedding_model.get_text_embedding(sentence)
print(f"embedding[:10]: {text_embedding[:10]}")

text_embeddings = embedding_model.get_text_embedding_batch([sentence, query])
print(f"text_embeddings[0][:10]: {text_embeddings[0][:10]}")
print(f"text_embeddings[1][:10]: {text_embeddings[1][:10]}")

query_embedding = embedding_model.get_query_embedding(query)
print(f"query_embedding[:10]: {query_embedding[:10]}")