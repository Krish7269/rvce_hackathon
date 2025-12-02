import json
import unittest

from code_generator import GeminiCodeGenerator, GenerationResult


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def generate_content(self, prompt: str):
        if self.calls >= len(self.outputs):
            raise AssertionError("No more fake outputs configured")
        text = self.outputs[self.calls]
        self.calls += 1
        return FakeResponse(text)


METADATA = {
    "columns": ["date", "value"],
    "types": {"date": "datetime", "value": "float"},
    "stats": {"value": {"mean": 10}},
    "row_count": 100,
}


class GeminiCodeGeneratorTests(unittest.TestCase):
    def _generator(self, outputs):
        return GeminiCodeGenerator(
            cache=None,
            max_retries=3,
            model=FakeModel(outputs),
        )

    def test_cache_hit_skips_model_call(self):
        code = "import pandas as pd\nprint('__RESULT__:{}')"
        generator = self._generator([code])
        result = generator.generate_analytics_code("test", METADATA, "summary")
        self.assertFalse(result.used_cache)
        cached = generator.generate_analytics_code("test", METADATA, "summary")
        self.assertTrue(cached.used_cache)
        self.assertEqual(code, cached.code)

    def test_retry_after_validation_error(self):
        bad_code = "exec('rm -rf /')"
        good_code = "import pandas as pd\nprint('__RESULT__:{}')"
        generator = self._generator([bad_code, good_code])
        result = generator.generate_analytics_code("test", METADATA, "summary")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.code, good_code)


if __name__ == "__main__":
    unittest.main()


