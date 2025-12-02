import io
import contextlib

def capture_output(code, local_env):
    """
    Executes dynamically generated code and captures stdout.
    'code' is a string containing Python code.
    'local_env' is a dict with variables (like df, plt).
    """

    f = io.StringIO()

    try:
        with contextlib.redirect_stdout(f):
            exec(code, local_env)
    except Exception as e:
        return f"ERROR: {str(e)}"

    return f.getvalue() if f.getvalue() else "Execution completed with no printed output."
