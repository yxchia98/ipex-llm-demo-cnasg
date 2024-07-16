from llama_index.llms.ipex_llm import IpexLLM

hf_model_path = "/llm-models/hf-models/gemma-2b-it"

saved_lowbit_model_path = "/llm-models/ipex-models/gemma-2b-it"

llm = IpexLLM.from_model_id(
    model_name=hf_model_path,
    tokenizer_name=hf_model_path,
)

llm._model.save_low_bit(saved_lowbit_model_path)
