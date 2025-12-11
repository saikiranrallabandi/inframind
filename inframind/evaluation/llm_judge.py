"""LLM-as-Judge for InfraMind evaluation."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


JUDGE_PROMPT = """You are an expert Infrastructure-as-Code reviewer evaluating code generation quality.

## Task
Instruction: {instruction}
Input: {input_text}

## Generated Response
{response}

## Evaluation Criteria
Score each dimension from 0-10:

1. **Syntax Correctness**: Is the code syntactically valid for its type (HCL/YAML/Dockerfile/etc)?
2. **Semantic Correctness**: Does the code do what was asked? Are resource types correct?
3. **Completeness**: Are all required components present? Nothing missing?
4. **Best Practices**: Does it follow security and industry standards?

## Output Format
Respond with ONLY valid JSON (no markdown, no explanation outside JSON):
{{
  "syntax_score": <0-10>,
  "semantic_score": <0-10>,
  "completeness_score": <0-10>,
  "best_practices_score": <0-10>,
  "overall_score": <0-10>,
  "would_work": <true or false>,
  "feedback": "<one sentence explanation>"
}}"""


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    syntax_score: float
    semantic_score: float
    completeness_score: float
    best_practices_score: float
    overall_score: float
    would_work: bool
    feedback: str
    raw_response: Optional[str] = None


class LLMJudge:
    """LLM-as-Judge for IaC evaluation using Claude or GPT-4."""

    def __init__(
        self,
        provider: str = "anthropic",  # "anthropic" or "openai"
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = provider

        if provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self.model = model or "claude-sonnet-4-20250514"
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        elif provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed. Run: pip install openai")
            self.model = model or "gpt-4"
            self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'")

    def evaluate(
        self,
        instruction: str,
        input_text: str,
        response: str
    ) -> JudgeResult:
        """Evaluate a single response."""

        prompt = JUDGE_PROMPT.format(
            instruction=instruction,
            input_text=input_text if input_text else "(none)",
            response=response
        )

        try:
            if self.provider == "anthropic":
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw_response = message.content[0].text
            else:  # openai
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                raw_response = completion.choices[0].message.content

            # Parse JSON response
            # Handle potential markdown code blocks
            json_str = raw_response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()

            result = json.loads(json_str)

            return JudgeResult(
                syntax_score=float(result.get("syntax_score", 0)),
                semantic_score=float(result.get("semantic_score", 0)),
                completeness_score=float(result.get("completeness_score", 0)),
                best_practices_score=float(result.get("best_practices_score", 0)),
                overall_score=float(result.get("overall_score", 0)),
                would_work=bool(result.get("would_work", False)),
                feedback=str(result.get("feedback", "")),
                raw_response=raw_response
            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {raw_response[:500]}")
            return JudgeResult(
                syntax_score=0, semantic_score=0, completeness_score=0,
                best_practices_score=0, overall_score=0, would_work=False,
                feedback=f"Parse error: {e}", raw_response=raw_response
            )
        except Exception as e:
            print(f"Evaluation error: {e}")
            return JudgeResult(
                syntax_score=0, semantic_score=0, completeness_score=0,
                best_practices_score=0, overall_score=0, would_work=False,
                feedback=f"Error: {e}", raw_response=None
            )

    def evaluate_batch(
        self,
        tasks: List[Dict],
        responses: List[str],
        verbose: bool = True
    ) -> List[JudgeResult]:
        """Evaluate batch of responses."""
        results = []

        for i, (task, response) in enumerate(zip(tasks, responses)):
            if verbose:
                print(f"  Evaluating {i+1}/{len(tasks)}: {task['category']}...", end=" ")

            result = self.evaluate(
                task["instruction"],
                task.get("input", ""),
                response
            )
            results.append(result)

            if verbose:
                print(f"Score: {result.overall_score}/10")

        return results
