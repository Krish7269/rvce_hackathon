"""Code generation engine for translating natural language analytics requests into
Pandas + Matplotlib Python scripts via Gemini."""

from __future__ import annotations

import ast
import json
import logging
import os
import textwrap
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - environment without SDK
    genai = None  # type: ignore


LOGGER = logging.getLogger(__name__)


def _normalise_query(query: str) -> str:
    return " ".join(query.strip().split()).lower()


class CodeCache:
    """Simple LRU cache keyed by prompt signature."""

    def __init__(self, max_size: int = 64) -> None:
        self.max_size = max_size
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: str) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)


class CodeValidator:
    """Validates generated Python to reduce runtime errors and security risks."""

    DISALLOWED_TOKENS: Tuple[str, ...] = (
        "os.remove",
        "os.rmdir",
        "shutil.rmtree",
        "subprocess",
        "requests",
        "open(",
        "__import__",
        "eval(",
        "exec(",
        "sys.exit",
    )

    def __init__(self) -> None:
        self.last_error: Optional[str] = None

    def validate(self, code: str) -> bool:
        self.last_error = None
        try:
            ast.parse(code)
        except SyntaxError as exc:  # pragma: no cover - branch validated via tests
            self.last_error = f"Syntax error: {exc}"
            return False

        lowered = code.lower()
        for token in self.DISALLOWED_TOKENS:
            if token in lowered:
                self.last_error = f"Disallowed token detected: {token}"
                return False
        return True


class PromptBuilder:
    """Constructs Gemini prompts with structured dataset context and constraints."""

    BASE_TEMPLATE = textwrap.dedent(
        """
        You are a senior Python data analyst. Convert the analytics request into a runnable Python script.

        Query Context:
        - Natural Language Goal: {user_query}
        - Dataset Summary: {dataset_summary}

        Dataset Metadata:
        - Columns: {columns}
        - Column Types: {column_types}
        - Descriptive Stats: {stats}
        - Row Count: {row_count}

        Output Requirements:
        1. Use ONLY pandas, numpy (optional), matplotlib, and seaborn.
        2. Call plt.switch_backend('Agg') and save figures to 'chart.png' (dpi=150, tight layout).
        3. Include descriptive comments explaining each analysis block.
        4. Define `CSV_PATH = "dataset.csv"` at the top and load the CSV once.
        5. Save any aggregated dataframe to 'analysis_output.csv' for downstream use.
        6. Print textual insights and end by printing a single line `__RESULT__:<JSON>` describing key numeric findings.
        7. Do NOT modify files, open network connections, or import unsafe modules.
        8. Provide a main() guard and call main() at the bottom.
        9. Output ONLY valid Python code, no markdown fences.

        Analysis Expectations:
        - Use pandas for filtering, aggregation, ranking, or correlation as needed.
        - Ensure visualizations include titles, axis labels, legends when applicable.
        - Handle ambiguous instructions with reasonable assumptions (comment them).
        - Always save figures even if visualization is optional.

        Generate the full Python script now:
        """
    ).strip()

    @staticmethod
    def format_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        columns = metadata.get("columns") or []
        column_types = metadata.get("types") or {}
        stats = metadata.get("stats") or {}
        row_count = metadata.get("row_count") or "unknown"
        return {
            "columns": json.dumps(columns),
            "column_types": json.dumps(column_types),
            "stats": json.dumps(stats)[:800],
            "row_count": str(row_count),
        }

    def build_prompt(
        self, user_query: str, metadata: Dict[str, Any], dataset_summary: str
    ) -> str:
        md = self.format_metadata(metadata)
        return self.BASE_TEMPLATE.format(
            user_query=user_query,
            dataset_summary=dataset_summary,
            columns=md["columns"],
            column_types=md["column_types"],
            stats=md["stats"],
            row_count=md["row_count"],
        )

    def refine_prompt(self, previous_prompt: str, error: str, attempt: int) -> str:
        refinement = textwrap.dedent(
            f"""
            Previous output failed because: {error}
            Please regenerate the entire script, strictly following every rule.
            This is retry #{attempt}. Guarantee syntactic correctness.
            """
        ).strip()
        return f"{previous_prompt}\n\n{refinement}"


@dataclass
class GenerationResult:
    code: str
    used_cache: bool = False
    attempts: int = 1


class GeminiCodeGenerator:
    """High-level interface for producing analytics scripts via Gemini."""

    def __init__(
        self,
        model_name: str = "models/gemini-1.5-pro",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        cache: Optional[CodeCache] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.max_retries = max_retries
        self.cache = cache or CodeCache()
        self.prompt_builder = PromptBuilder()
        self.validator = CodeValidator()

        if model is not None:
            self.model = model
        else:
            if genai is None:
                raise ImportError(
                    "google-generativeai is required unless a custom model is supplied."
                )
            api_key = api_key or os.getenv("AIzaSyB7jV6fxUzArZC2vJRmjYdZdX3ZWy3FqIU")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY is required for code generation.")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)

    def _cache_key(
        self, query: str, metadata: Dict[str, Any], dataset_summary: str
    ) -> str:
        return json.dumps(
            {
                "query": _normalise_query(query),
                "columns": metadata.get("columns"),
                "types": metadata.get("types"),
                "summary": dataset_summary.strip(),
            },
            sort_keys=True,
        )

    def _extract_code(self, response: Any) -> str:
        text = getattr(response, "text", None) or ""
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            pieces = text.split("\n", 1)
            text = pieces[1] if len(pieces) > 1 else ""
        return text.strip()

    def generate_analytics_code(
        self, user_query: str, metadata: Dict[str, Any], dataset_summary: str
    ) -> GenerationResult:
        cache_key = self._cache_key(user_query, metadata, dataset_summary)
        cached = self.cache.get(cache_key)
        if cached:
            LOGGER.debug("Cache hit for query '%s'", user_query)
            return GenerationResult(code=cached, used_cache=True, attempts=0)

        prompt = self.prompt_builder.build_prompt(user_query, metadata, dataset_summary)
        last_error = "initial attempt"
        for attempt in range(1, self.max_retries + 1):
            LOGGER.debug("Gemini generation attempt %s/%s", attempt, self.max_retries)
            response = self.model.generate_content(prompt)
            code = self._extract_code(response)

            if self.validator.validate(code):
                self.cache.set(cache_key, code)
                return GenerationResult(code=code, attempts=attempt)

            last_error = self.validator.last_error or "unknown validation issue"
            prompt = self.prompt_builder.refine_prompt(prompt, last_error, attempt + 1)
            LOGGER.warning(
                "Validation failed on attempt %s: %s", attempt, last_error
            )

        raise ValueError(f"Failed to generate valid code after retries: {last_error}")


def generate_analytics_code(
    user_query: str, csv_metadata: Dict[str, Any], dataset_summary: str
) -> GenerationResult:
    """Convenience wrapper for external callers."""
    generator = GeminiCodeGenerator()
    return generator.generate_analytics_code(user_query, csv_metadata, dataset_summary)


