import asyncio
import re
import requests
from typing import List

import httpx
import torch
import tqdm
from transformers import AutoTokenizer, LlamaTokenizer

from lm_eval.base import BaseLM


class lightllm(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="huggyllama/llama-7b",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_8bit=True,
    ):
        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained)
        self.api_url = "http://localhost:8000/generate"

        self.batch_size_per_gpu = 128
        print("eos_token: ", self.tokenizer.eos_token)
        print("Using framework lightllm")

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 128000
        # try:
        #     return self.model.config.n_ctx
        # except AttributeError:
        #     return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        raise NotImplementedError
        with torch.no_grad():
            return self.model(inps)[0]

    async def async_run(self, prompt: str, **kwargs) -> str:
        headers = {'Content-Type': 'application/json'}
        pload = {
            'inputs': prompt,
            "parameters": {
                'do_sample': False,
                'ignore_eos': False,
                'max_new_tokens': self.max_gen_toks,
            }
        }
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(self.api_url, headers=headers, json=pload)
        text = response.json()['generated_text'][0]
        return text
    
    async def _async_run_batch(self, prompts: List[str], **kwargs) -> List[str]:
        tasks: List[asyncio.Task] = [self.async_run(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def run_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return asyncio.run(self._async_run_batch(prompts, **kwargs))

    def _model_generate(self, context: torch.Tensor, max_length, eos_token_id):
        prompt = self.tok_decode(context.tolist()[0])
        headers = {'Content-Type': 'application/json'}
        pload = {
            'inputs': prompt,
            "parameters": {
                'do_sample': False,
                'ignore_eos': False,
                'max_new_tokens': max_length - len(context[0]) ,
            }
        }
        response = requests.post(self.api_url, headers=headers, json=pload, stream=False)
        text = response.json()['generated_text'][0]
        # print(prompt)
        # print(text)
        return torch.tensor([context.tolist()[0] + self.tok_encode(text)])

    def greedy_until(self, reqs):
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        res = []
        real_match = re.compile(r'^(.*?#### \d+(\.\d+)?)', re.DOTALL)

        # print(reqs)

        prompts = [req[0] for req in reqs]
        texts = self.run_batch(prompts)

        for text in texts:

            target = real_match.search(text)
            if target is not None:
                text = target.group(1)
            else:
                print("Warning: no match found for", text)

            res.append(text)

        return res
