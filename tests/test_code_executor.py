import importlib.util
import json
import textwrap
import unittest

from code_executor import CodeExecutor


SUCCESS_CODE = textwrap.dedent(
    """
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    CSV_PATH = "dummy.csv"

    def main():
        plt.switch_backend('Agg')
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
        summary = {'mean': df['y'].mean()}
        ax = df.plot(x='x', y='y', kind='line', title='Trend')
        ax.set_ylabel('y')
        plt.tight_layout()
        plt.savefig('chart.png', dpi=150, bbox_inches='tight')
        df.to_csv('analysis_output.csv', index=False)
        print("__RESULT__:" + json.dumps(summary))

    if __name__ == "__main__":
        main()
    """
).strip()


ERROR_CODE = "raise ValueError('bad data')\nprint('__RESULT__:{}')"


TIMEOUT_CODE = textwrap.dedent(
    """
    import time
    print("starting")
    time.sleep(5)
    print("__RESULT__:{}")
    """
).strip()


class CodeExecutorTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("pandas") and importlib.util.find_spec("matplotlib"),
        "pandas and matplotlib are required for this test",
    )
    def test_execute_analysis_code_success(self):
        executor = CodeExecutor(timeout=10)
        result = executor.execute_analysis_code(SUCCESS_CODE)
        self.assertTrue(result.success, result.error)
        self.assertTrue(result.visualization.endswith("chart.png"))
        self.assertIn("mean", result.data)

    def test_execute_analysis_code_handles_error(self):
        executor = CodeExecutor(timeout=5)
        result = executor.execute_analysis_code(ERROR_CODE)
        self.assertFalse(result.success)
        self.assertIn("ValueError", result.error or "")

    def test_execute_analysis_code_timeout(self):
        executor = CodeExecutor(timeout=1)
        result = executor.execute_analysis_code(TIMEOUT_CODE)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error or "")


if __name__ == "__main__":
    unittest.main()


