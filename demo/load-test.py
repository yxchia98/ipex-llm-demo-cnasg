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
    parser = argparse.ArgumentParser(description='IPEX-LLM load and inference example.')
    parser.add_argument('--load-path', type=str, default=None,
                        help='The path to load the low-bit model.')
    args = parser.parse_args()
    load_path = args.load_path

    if load_path:
        model = AutoModelForCausalLM.load_low_bit(load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path)
    else:
        print('please provide IPEX-LLM model path')
        exit()

    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=64)
    input_str = "What is AI?"
    output = pipeline(input_str)[0]["generated_text"]
    print(f"Prompt: {input_str}")
    print(f"Output: {output}")
