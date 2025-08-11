from typing import Any, List, Optional
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import torch
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class PresencePenaltyProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for sequence in input_ids:
            for token_id in set(sequence.tolist()):
                scores[:, token_id] -= self.penalty
        return scores


class MyLLM(LLM):
    model_name: str = "/tmp/pycharm_project_581/Qwen3-8B"
    max_new_tokens: int = 32768
    temperature: float = 0.6
    enable_thinking:bool = False
    # max_total_tokens: int = 32768  # Qwen-3-8B 支持的最大上下文长度
    device_map: str = 'auto'
    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    presence_penalty_value:float = 0

    def __init__(self, **kwargs):
        super().__init__()
        self.presence_penalty_value = kwargs.get("presence_penalty", 0.0)
        self.enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        self.model_name: str = kwargs.get("model_name", self.model_name)
        self.device_map = kwargs.get("device_map", self.device_map)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype='auto',
            device_map=self.device_map,
            # load_in_8bit=True,
            # use_flash_attention_2=True,
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
        self.presence_penalty_value = kwargs.get("presence_penalty", 0.0)
        logits_processor = LogitsProcessorList()
        if self.presence_penalty_value != 0:
            logits_processor.append(PresencePenaltyProcessor(self.presence_penalty_value))
        messages = [{"role": "user", "content": prompt}]
        self.enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        # 统计输入 tokens
        prompt_token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        prompt_token_count = len(prompt_token_ids)
        print(f"\n[Token Usage] Prompt tokens: {prompt_token_count} / {self.max_new_tokens}")

        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)


        if model_inputs['input_ids'].max() >= self._tokenizer.vocab_size:
            model_inputs['input_ids'] = torch.clamp(
                model_inputs['input_ids'],
                0,
                self._tokenizer.vocab_size - 1
            )

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            logits_processor=logits_processor

        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_token_count = len(output_ids)
        total_token_count = prompt_token_count + output_token_count

        print(f"[Token Usage] Output tokens: {output_token_count}")
        print(f"[Token Usage] Total tokens: {total_token_count} / {self.max_new_tokens}\n")
        # if enable_thinking:
        #     try:
        #         # 解析 thinking content（假设151668是</think>的token ID）
        #         index = len(output_ids) - output_ids[::-1].index(151668)
        #         thinking_content = self._tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        #         content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        #         return f"[思考过程]\n{thinking_content}\n\n[回答]\n{content}"
        #     except ValueError:
        #         # 如果没有找到thinking标记，返回完整内容
        #         content = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        #         return f"{content}"
        # else:
            # 直接返回完整回答
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print(content)
        return content

# 可选封装函数
def load_llm(enable_thinking) ->  MyLLM:
    return MyLLM(enable_thinking=enable_thinking)