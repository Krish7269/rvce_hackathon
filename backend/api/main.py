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
    2. Passes the question + data to code_generator and code_executor.
    3. Returns standardized JSON: text, table, chart
    """
    # Load DataFrame
    try:
        df = df_from_upload(file)
    except Exception as e:
        return JSONResponse({"text": f"Failed to read file: {e}"}, status_code=400)

    generated_code = None

    # -----------------------------------
    # Step 1 — Call code_generator (if available)
    # -----------------------------------

    if generator_fn is not None:
        try:
            # Try multiple calling conventions
            try:
                generated_code = generator_fn(df=df, question=question)
            except TypeError:
                try:
                    generated_code = generator_fn(question)
                except TypeError:
                    generated_code = generator_fn()
        except Exception as e:
            tb = traceback.format_exc()
            return JSONResponse({"text": f"Error in code_generator:\n{tb}"}, status_code=500)

    # -----------------------------------
    # Step 2 — Call code_executor
    # -----------------------------------

    result = None

    if executor_fn is not None:
        attempts = [
            {"df": df, "code": generated_code, "question": question},
            {"df": df, "question": question},
            {"question": question},
            {"df": df},
            {"code": generated_code},
        ]

        for args in attempts:
            try:
                clean_args = {k: v for k, v in args.items() if v is not None}
                result = executor_fn(**clean_args)
                break
            except TypeError:
                continue
            except Exception:
                tb = traceback.format_exc()
                return JSONResponse({"text": f"Executor error:\n{tb}"}, status_code=500)

        if result is None:
            return JSONResponse(
                {"text": "Executor function found but could not be called with any signature."},
                status_code=500
            )

    else:
        # -----------------------------------
        # Fallback: run generated code as script
        # -----------------------------------
        if generated_code is None:
            return JSONResponse(
                {"text": "No executor or generated code available to run."},
                status_code=500
            )

        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
                tf.write(generated_code)
                script_path = tf.name

            data_path = script_path + ".csv"
            df.to_csv(data_path, index=False)

            import subprocess, sys
            proc = subprocess.run(
                [sys.executable, script_path, data_path, question],
                capture_output=True,
                text=True,
                timeout=60
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            import json
            try:
                result = json.loads(stdout)
            except:
                result = {"text": stdout or stderr}

        except Exception:
            tb = traceback.format_exc()
            return JSONResponse({"text": f"Error running generated code:\n{tb}"}, status_code=500)

        finally:
            try:
                os.remove(script_path)
                os.remove(data_path)
            except:
                pass

    # -----------------------------------
    # Step 3 — Normalize result into text/table/chart
    # -----------------------------------

    out_text = None
    out_table = None
    out_chart = None

    if isinstance(result, str):
        out_text = result

    elif isinstance(result, dict):
        out_text = result.get("text") or result.get("summary") or ""

        table = (
            result.get("table") or
            result.get("data") or
            result.get("df")
        )
        if table is not None:
            if hasattr(table, "to_dict"):
                out_table = table.to_dict(orient="records")
            elif isinstance(table, list):
                out_table = table

        chart = result.get("chart")
        if chart is not None:
            if hasattr(chart, "savefig"):
                out_chart = chart_to_base64_bytes(chart)
            elif isinstance(chart, str) and chart.startswith("data:image"):
                out_chart = chart

    elif hasattr(result, "to_dict"):
        out_table = result.to_dict(orient="records")
        out_text = f"Returned table with {len(out_table)} rows."

    else:
        out_text = str(result)

    return {
        "text": out_text or "",
        "table": out_table,
        "chart": out_chart
    }
