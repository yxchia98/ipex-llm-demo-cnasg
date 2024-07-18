from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen2-1.5B-Instruct", repo_type="model", local_dir="./llm-models/hf-models/Qwen2-1.5B-Instruct")

snapshot_download(repo_id="BAAI/bge-small-en-v1.5", repo_type="model", local_dir="./llm-models/hf-models/bge-small-en-v1.5")


# snapshot_download(repo_id="yahma/alpaca-cleaned", repo_type="dataset", local_dir=".")