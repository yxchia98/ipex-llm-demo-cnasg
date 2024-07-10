# IPEX LLM MODEL
import warnings
from llama_index.llms.ipex_llm import IpexLLM

# Transform a string into input zephyr-specific input
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



warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*padding_mask.*"
)



# llm = IpexLLM.from_model_id(
#     model_name="/llm-models/hf-models/Phi-3-mini-4k-instruct",
#     tokenizer_name="/llm-models/hf-models/Phi-3-mini-4k-instruct",
#     context_window=512,
#     max_new_tokens=128,
#     generate_kwargs={"do_sample": False},
#     completion_to_prompt=completion_to_prompt,
#     messages_to_prompt=messages_to_prompt,
# )


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


# #Text Completion
# completion_response = llm.complete("Once upon a time, ")
# print(completion_response.text)


# # Streaming Text Completion
# response_iter = llm.stream_complete("What is AI?")
# for response in response_iter:
#     print(response.delta, end="", flush=True)


# # Chat
# from llama_index.core.llms import ChatMessage
# message = ChatMessage(role="user", content="Explain Big Bang Theory briefly")
# resp = llm.chat([message])
# print(resp)

# Streaming Chat
from llama_index.core.llms import ChatMessage
message = ChatMessage(role="user", content="What is AI?")
resp = llm.stream_chat([message], max_tokens=256)
for r in resp:
    print(r.delta, end="")