from typing import override

from pydantic_ai import Agent

from govec_bench.adapters.base import QueryResult
from govec_bench.generation.base import Generator

SYSTEM_PROMPT = (
    "Answer the question using only the numbered sources below. Cite sources inline "
    "like [1], [2]. If the sources don't answer the question, say so plainly."
)


class PydanticAIGenerator(Generator, name="pydantic-ai"):
    def __init__(self, model: str) -> None:
        self._agent = Agent(model, system_prompt=SYSTEM_PROMPT)

    @override
    def generate(self, question: str, chunks: list[QueryResult]) -> str:
        sources = "\n\n".join(
            f"[{i}] {(chunk.metadata or {}).get('heading', '')}\n{(chunk.metadata or {}).get('text', '')}"
            for i, chunk in enumerate(chunks, start=1)
        )
        prompt = f"Question: {question}\n\nSources:\n{sources}"

        result = self._agent.run_sync(prompt)
        return result.output
