"""Local-model backend: an in-process vLLM engine (no external server process).

The engine is a per-process singleton. Fine-tuning integration points:
  * load_adapter(name, path): register/replace a LoRA adapter; subsequent calls
    with lora="name" use it.
  * sleep()/wake(): release GPU memory so a co-located training step can run.
"""

import os
import threading

from . import config


class LocalVLLM:
    _lock = threading.Lock()
    _singleton = None

    @classmethod
    def instance(cls, alias_cfg=None):
        with cls._lock:
            if cls._singleton is None:
                cls._singleton = cls(alias_cfg or config.LOCAL_MODELS[config.DEFAULT_LOCAL_ALIAS])
            return cls._singleton

    def __init__(self, cfg):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.LOCAL_GPU)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM

        # LoRA hot-swap needs Triton LoRA kernels, which fail to compile on
        # some older GPUs (e.g. Turing SM7.5); opt in via LLM_RUNTIMES_LOCAL_LORA=1.
        self.lora_enabled = os.environ.get("LLM_RUNTIMES_LOCAL_LORA", "0") == "1"
        kwargs = dict(
            model=cfg["hf_id"],
            dtype=cfg.get("dtype", "float16"),
            max_model_len=cfg.get("max_model_len", 16384),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            enable_lora=self.lora_enabled,
            enforce_eager=True,  # saves memory on 11GB cards
        )
        if cfg.get("quantization"):
            kwargs["quantization"] = cfg["quantization"]
        self.llm = LLM(**kwargs)
        self.cfg = cfg
        self._adapters = {}  # name -> (int_id, path)
        self._next_adapter_id = 1
        self._gen_lock = threading.Lock()  # vLLM offline engine is not thread-safe

    # ---- generation -------------------------------------------------------
    def chat(self, messages, temperature=0.7, max_tokens=2048, n=1, stop=None,
             lora=None):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sp = SamplingParams(
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens or 2048,
            n=n,
            stop=stop,
        )
        lora_req = None
        if lora and lora in self._adapters:
            aid, path = self._adapters[lora]
            lora_req = LoRARequest(lora, aid, path)
        with self._gen_lock:
            try:
                # Disable reasoning blocks on models that support the switch
                # (e.g. Qwen3), so scaffolds get clean assistant text.
                outs = self.llm.chat(
                    messages, sp, lora_request=lora_req, use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                outs = self.llm.chat(messages, sp, lora_request=lora_req,
                                     use_tqdm=False)
        return [o.text for o in outs[0].outputs]

    # ---- fine-tuning control ---------------------------------------------
    def load_adapter(self, name, path):
        if not self.lora_enabled:
            raise RuntimeError(
                "LoRA is disabled (set LLM_RUNTIMES_LOCAL_LORA=1; requires a GPU "
                "where Triton LoRA kernels compile, i.e. Ampere or newer)"
            )
        self._adapters[name] = (self._next_adapter_id, path)
        self._next_adapter_id += 1
        return {"name": name, "path": path}

    def adapters(self):
        return {k: v[1] for k, v in self._adapters.items()}

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake(self):
        self.llm.wake_up()
