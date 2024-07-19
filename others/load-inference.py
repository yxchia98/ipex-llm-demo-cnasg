#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, TextGenerationPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer save_load example')
    parser.add_argument('--load-path', type=str, default=None,
                        help='The path to load the low-bit model.')
    parser.add_argument('--prompt', type=str, default="AI是什么？",
                        help='Prompt to infer') 
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    args = parser.parse_args()
    load_path = args.load_path
    prompt = args.prompt
    n_predict = args.n_predict

    if load_path:
        model = AutoModelForCausalLM.load_low_bit(load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path)
    else:
        print("No --load-path specified!")

    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=n_predict)
    # input_str = prompt

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ]
    input_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
    
    output = pipeline(input_str)[0]["generated_text"]
    print(f"Prompt: {input_str}")
    print(f"Output: {output}")

    # save_path = args.save_path
    # if save_path:
    #     model.save_low_bit(save_path)
    #     tokenizer.save_pretrained(save_path)
    #     print(f"Model and tokenizer are saved to {save_path}")
