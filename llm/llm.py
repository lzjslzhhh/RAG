from typing import Any, List, Optional

from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class RemoteLLM(LLM):
    model_name: str = "/tmp/pycharm_project_581/Qwen3-8B"
    max_new_tokens: int = 32768
    temperature: float = 0.6

    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype='auto',
            device_map="auto",
            trust_remote_code=True
        )

    @property
    def _llm_type(self) -> str:
        return "qwen-local"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # 解析 thinking content
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token ID
        except ValueError:
            index = 0

        thinking_content = self._tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return f"[思考过程]\n{thinking_content}\n\n[回答]\n{content}"

# 可选封装函数
def load_llm() -> RemoteLLM:
    return RemoteLLM()