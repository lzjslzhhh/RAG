from typing import Any, List, Optional

from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class MyLLM(LLM):
    model_name: str = "/tmp/pycharm_project_581/Qwen3-8B"
    max_new_tokens: int = 32768
    temperature: float = 0.6
    enable_thinking:bool = False

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
            device_map="cuda",
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
        enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        if enable_thinking:
            try:
                # 解析 thinking content（假设151668是</think>的token ID）
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self._tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                return f"[思考过程]\n{thinking_content}\n\n[回答]\n{content}"
            except ValueError:
                # 如果没有找到thinking标记，返回完整内容
                content = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                return f"[回答]\n{content}"
        else:
            # 直接返回完整回答
            content = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            return content

# 可选封装函数
def load_llm() ->  MyLLM:
    return MyLLM()