from typing import Any, Dict, List

from langchain.callbacks import StdOutCallbackHandler
from langchain.schema import LLMResult

class RichStdOutCallbackHandler(StdOutCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        for prompt in prompts:
            print(f"Prompt:\n{prompt}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generation in response.generations:
            print(f"Response:\n{generation.text}\n\n")
