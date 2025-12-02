# backend/api/main.py

import io
import os
import tempfile
import base64
import traceback
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

import pandas as pd
import importlib

# Extra imports to reuse the notebook-style pipeline
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

try:  # pragma: no cover - optional in some environments
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover
    genai = None

from agent.utils import capture_output  # type: ignore
from agent.code_generator import CodeGenerator  # type: ignore
from agent.explanation_agent import ExplanationAgent  # type: ignore


def _load_local_env(path: str = ".emv") -> None:
    """
    Lightweight .env loader so you can keep API keys in the `.emv` file.

    Format (one per line, '#' comments allowed):
        GEMINI_API_KEY=...
        GOOGLE_API_KEY=...
    """
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Fail silently – missing keys will be reported later in a clear error.
        return


# Load `.emv` once when the backend starts so os.getenv() can see the keys.
_load_local_env()

app = FastAPI(title="AI-Analyzer API Wrapper")

# ------------------------------------------------------
#  Helper: Try to import functions from code_generator
# ------------------------------------------------------

def try_import(module_name: str, candidates: list):
    """
    Try to import module_name and return the first callable attribute from candidates found.
    Returns (callable, attr_name) or (None, None).
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None, None

    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn, name
    return None, None


GENERATOR_FN_CANDIDATES = ["generate_code", "generate", "generate_analysis", "main_generate"]
EXECUTOR_FN_CANDIDATES = ["execute_code", "execute", "run_code", "run", "main_execute"]

generator_fn, generator_name = try_import("code_generator", GENERATOR_FN_CANDIDATES)
executor_fn, executor_name = try_import("code_executor", EXECUTOR_FN_CANDIDATES)


@app.get("/")
def root():
    return {
        "status": "running",
        "generator_detected": generator_name,
        "executor_detected": executor_name
    }


# ------------------------------------------------------
#  File handling helpers
# ------------------------------------------------------

def df_from_upload(file: UploadFile) -> pd.DataFrame:
    """Convert uploaded CSV, Excel, or JSON into a pandas DataFrame."""
    content = file.file.read()
    fname = file.filename.lower()

    if fname.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif fname.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    elif fname.endswith(".json"):
        return pd.read_json(io.BytesIO(content))
    else:
        raise ValueError(f"Unsupported file type: {fname}")


def chart_to_base64_bytes(fig):
    """Convert matplotlib figure to base64 PNG string."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ------------------------------------------------------
#  Main analyze endpoint
# ------------------------------------------------------

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    API endpoint:
    1. Loads the uploaded file into a pandas DataFrame.
    2. Uses the same Gemini + agent pipeline as `core_agent.ipynb` to
       generate and execute analysis code.
    3. Returns standardized JSON: text, table, chart
    """
    # Load DataFrame
    try:
        df = df_from_upload(file)
    except Exception as e:
        return JSONResponse({"text": f"Failed to read file: {e}"}, status_code=400)

    # --------------------------------------------------
    # 1) Initialise LLM & notebook-style agents
    # --------------------------------------------------
    if genai is None:
        return JSONResponse(
            {"text": "google-generativeai is not installed. Install `google-generativeai` and set GEMINI_API_KEY."},
            status_code=500,
        )

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GENAI_API_KEY")
    )
    if not api_key:
        return JSONResponse(
            {"text": "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or `.emv`."},
            status_code=500,
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
    except Exception:
        tb = traceback.format_exc()
        return JSONResponse({"text": f"Failed to initialise Gemini model:\n{tb}"}, status_code=500)

    def llm(prompt: str) -> str:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or ""

    code_gen = CodeGenerator(llm)
    exp_agent = ExplanationAgent(llm)

    # --------------------------------------------------
    # 2) Generate Python code for this question + dataset
    # --------------------------------------------------
    try:
        raw_code = code_gen.generate(question, df)
    except Exception:
        tb = traceback.format_exc()
        return JSONResponse({"text": f"Error in code generation:\n{tb}"}, status_code=500)

    # Basic cleanup: strip markdown fences if the model added them
    cleaned_code = raw_code.replace("```python", "").replace("```", "")

    # --------------------------------------------------
    # 3) Execute the code with df / plotting libraries available
    # --------------------------------------------------
    local_env = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
    }

    # Ensure any previous plot file does not leak between requests
    plot_path = "output_plot.png"
    if os.path.exists(plot_path):
        try:
            os.remove(plot_path)
        except OSError:
            pass

    try:
        exec_output = capture_output(cleaned_code, local_env)
    except Exception:
        tb = traceback.format_exc()
        return JSONResponse({"text": f"Error while executing generated code:\n{tb}"}, status_code=500)

    # --------------------------------------------------
    # 4) Summarise Python output and get natural-language explanation
    # --------------------------------------------------
    python_summary = (exec_output or "")[:500]

    try:
        explanation = exp_agent.explain(question, python_summary)
    except Exception:
        tb = traceback.format_exc()
        return JSONResponse({"text": f"Error in explanation agent:\n{tb}"}, status_code=500)

    # --------------------------------------------------
    # 5) Build a simple result structure for the frontend
    # --------------------------------------------------
    return {
        "text": explanation or exec_output or "Analysis completed.",
        # Provide a small preview of the dataset back to the UI.
        "table": df.head(20).to_dict(orient="records"),
        # For now we don't try to parse or embed the plot; the Streamlit
        # app will still display text + table, which is the critical bit.
        "chart": None,
    }
