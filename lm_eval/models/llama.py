import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from lm_eval.base import BaseLM

from lm_eval.models.modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config, change_llama_heavy_const
from lm_eval.models.h2o_llama import convert_h2o, reset_h2o
from lm_eval.models.sparq_llama import convert_sparq, change_sparq_para


class LlamaLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="huggyllama/llama-7b",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_8bit=False,
        sparse_mode="ds",
        # sparse_mode="h2o",
        # sparse_mode="sparq",
        # sparse_mode=None,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        assert sparse_mode in [None, "ds", "h2o", "sparq"], f"Invalid sparse mode {sparse_mode}"
        self.sparse_mode = sparse_mode

        self.batch_size_per_gpu = batch_size

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        if load_8bit:
            self.model = LlamaForCausalLM.from_pretrained(
                pretrained, revision=revision, device_map="auto", load_in_8bit=True
            )
        else:
            print(pretrained, revision)
            self.model = LlamaForCausalLM.from_pretrained(
                    pretrained, revision=revision, torch_dtype=torch.float16
                ).to(self._device)
            

        config = LlamaConfig.from_pretrained(pretrained, revision=revision)
        if self.sparse_mode == "ds":
            self.model = convert_kvcache_llama_heavy_recent(self.model, config, 128, 4, 4)
            channel_path = "/home/ec2-user/DoubleSparse/llama2-7b-chat-qk-channel-config.json"
            channel_config = None
            with open(channel_path, "r") as f:
                channel_config = json.load(f)
            self.model = convert_llama_channel_config(self.model, channel_config, "qk")
        elif self.sparse_mode == "h2o":
            config.heavy_ratio = 0.25
            config.recent_ratio = 0.1
            self.model = convert_h2o(self.model, config)
        elif self.sparse_mode == "sparq":
            k = 128
            r = 16
            self.model = convert_sparq(self.model, config, k, r)

        self.model.eval()

        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
        )
        self.vocab_size = len(self.tokenizer)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        # print(inps.shape)
        if self.sparse_mode == "h2o":
            self.model = reset_h2o(self.model)
        elif self.sparse_mode == "ds":
            self.model = change_llama_heavy_const(self.model, inps.shape[-1] // 16 ,4,4)
        elif self.sparse_mode == "sparq":
            self.model = change_sparq_para(self.model, inps.shape[-1] // 8, 16)
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        if self.sparse_mode == "h2o":
            self.model = reset_h2o(self.model)
        elif self.sparse_mode == "ds":
            self.model = change_llama_heavy_const(self.model, context.shape[-1] // 16 ,4,4)
        elif self.sparse_mode == "sparq":
            self.model = change_sparq_para(self.model, context.shape[-1] // 8, 16)
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
