"""Strategy proposers: random mutation and optional LLM-based."""
from __future__ import annotations
import json
import logging
import os
from abc import ABC, abstractmethod
import numpy as np
from evolveclaw_ramsey.ramsey.strategies import (
    Strategy, RandomStrategy, PaleyStrategy, CyclicStrategy, PerturbedStrategy, strategy_from_dict,
)
from evolveclaw_ramsey.utils.logging import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

class Proposer(ABC):
    @abstractmethod
    def propose(self, parents: list[Strategy], scores: list[float], problem: dict) -> Strategy:
        ...

class RandomMutationProposer(Proposer):
    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def propose(self, parents: list[Strategy], scores: list[float], problem: dict) -> Strategy:
        if not parents:
            return RandomStrategy(edge_prob=self._rng.random(), rng=self._rng)
        parent = parents[0]
        if self._rng.random() < 0.15:
            n = problem.get("n", 17)
            choice = self._rng.integers(0, 4)
            if choice == 0:
                return RandomStrategy(edge_prob=self._rng.random(), rng=self._rng)
            elif choice == 1:
                return PaleyStrategy(rng=self._rng)
            elif choice == 2:
                num_offsets = int(self._rng.integers(1, max(2, n // 2)))
                offsets = [int(x) for x in self._rng.integers(1, n, size=num_offsets)]
                return CyclicStrategy(offsets=offsets, rng=self._rng)
            else:
                return PerturbedStrategy(base=parent, flip_prob=float(self._rng.uniform(0.01, 0.2)), rng=self._rng)
        return parent.mutate(self._rng)

class LLMProposer(Proposer):
    def __init__(self, provider: str, model: str, api_key: str, rng: np.random.Generator):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._rng = rng
        self._fallback = RandomMutationProposer(rng=rng)

    def propose(self, parents: list[Strategy], scores: list[float], problem: dict) -> Strategy:
        prompt = self._build_prompt(parents, scores, problem)
        try:
            response_text = self._call_llm(prompt)
            return self._parse_response(response_text)
        except Exception as e:
            logger.warning(f"LLM proposer failed, falling back to random: {e}")
            return self._fallback.propose(parents, scores, problem)

    def _build_prompt(self, parents, scores, problem):
        parent_info = ""
        for p, s in zip(parents, scores):
            parent_info += f"  Strategy: {json.dumps(p.to_dict())}\n  Score: {s}\n\n"
        return f"""You are optimizing graph construction strategies for Ramsey number R({problem['s']},{problem['t']}).
Goal: Find a graph on {problem['n']} vertices with NO clique of size {problem['s']} and NO independent set of size {problem['t']}.
Higher score = better. Perfect score = {problem['n']} (zero violations).

Current parent strategies:
{parent_info}
Available strategy types and their JSON format:
1. {{"name": "random", "edge_prob": <float 0-1>}}
2. {{"name": "paley"}}
3. {{"name": "cyclic", "offsets": [<int>, ...]}}
4. {{"name": "perturbed", "base": <strategy dict>, "flip_prob": <float 0-0.5>}}

Suggest an improved strategy. Return ONLY a JSON object with the strategy specification, nothing else."""

    def _call_llm(self, prompt):
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(model=self.model, max_tokens=500,
                                            messages=[{"role": "user", "content": prompt}])
            return message.content[0].text
        elif self.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(model=self.model, max_tokens=500,
                                                     messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _parse_response(self, text):
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        d = json.loads(text)
        return strategy_from_dict(d, self._rng)

def create_proposer(config: dict, rng: np.random.Generator) -> Proposer:
    proposer_type = config.get("type", "random")
    if proposer_type == "random":
        return RandomMutationProposer(rng=rng)
    elif proposer_type == "llm":
        provider = config.get("llm_provider", "anthropic")
        model = config.get("llm_model", "claude-sonnet-4-20250514")
        api_key_env = config.get("llm_api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"LLM proposer requires API key. Set the {api_key_env} environment variable.")
        return LLMProposer(provider=provider, model=model, api_key=api_key, rng=rng)
    else:
        raise ValueError(f"Unknown proposer type: {proposer_type}")
