# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# --- Optional: import your modules (fall back if not present) ---
try:
    from agent.code_generator import CodeGenerator
    from agent.explanation_agent import ExplanationAgent
    from agent.utils import capture_output
    IMPORTED_AGENT_MODULES = True
except Exception as e:
    IMPORTED_AGENT_MODULES = False
    # simple fallback capture_output
    def capture_output(code, local_env):
        import contextlib, sys, io
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, local_env)
        except Exception as ex:
            return f"ERROR: {str(ex)}"
        return buf.getvalue()

try:
    from agents.report_generator import ReportGenerator
    report_gen = ReportGenerator()
    IMPORTED_REPORT = True
except Exception as e:
    IMPORTED_REPORT = False
    # fallback minimal PDF generator using fpdf
    from fpdf import FPDF
    class ReportGenerator:
        def __init__(self):
            self.pdf = FPDF()
            self.pdf.set_auto_page_break(auto=True, margin=15)
        def generate_report(self, title, summary, images=[]):
            self.pdf = FPDF()
            self.pdf.set_auto_page_break(auto=True, margin=15)
            self.pdf.add_page()
            self.pdf.set_font("Arial", "B", 16)
            self.pdf.cell(0, 10, title, ln=True, align="C")
            self.pdf.ln(6)
            self.pdf.set_font("Arial", size=11)
            self.pdf.multi_cell(0, 8, summary)
            self.pdf.ln(6)
            for img in images:
                if os.path.exists(img) and os.path.getsize(img) > 2048:
                    try:
                        self.pdf.image(img, w=170)
                        self.pdf.ln(6)
                    except Exception as e:
                        self.pdf.multi_cell(0,8,f"[Error adding image {img}: {e}]")
                else:
                    self.pdf.multi_cell(0,8,f"[Image missing or too small: {img}]")
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = f"report_{now}.pdf"
            self.pdf.output(out)
            return out
    report_gen = ReportGenerator()

# --- Google Gemini setup (UI for API key) ---
import google.generativeai as genai

st.set_page_config(page_title="AI Analyst Dashboard", layout="wide")

st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("Google API key (Gemini)", type="password")
model_choice = st.sidebar.selectbox("Model", ["gemini-1.5-flash","gemini-1.5-pro","gemini-2.5-flash"], index=0)

if api_key_input:
    genai.configure(api_key=api_key_input)
    model = genai.GenerativeModel(model_choice)
    st.sidebar.success("API key configured")
else:
    model = None
    st.sidebar.info("Enter API key to enable LLM (or keep running for local fallback)")

# --- UI layout ---
st.title("AI Analyst — Advanced Dashboard")
st.markdown("Upload sales CSV, ask questions (NLP), get plots, explanations and downloadable PDF reports.")

# File upload
st.sidebar.subheader("Upload / Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
drive_link = st.sidebar.text_input("Or paste Google Drive CSV link (public)")

# optional sample queries
st.sidebar.subheader("Suggested queries")
suggested = st.sidebar.selectbox("Pick a suggested query",
                                 ["Show total revenue per month", "Top 5 products by revenue",
                                  "Revenue by region", "Forecast next 2 months revenue", "Show units sold trend"])
use_suggestion = st.sidebar.button("Use suggestion")

if use_suggestion:
    default_query = suggested
else:
    default_query = ""

# show dataset
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif drive_link:
    try:
        # support raw Google Drive link or direct csv link
        if "drive.google.com" in drive_link and "export=download" not in drive_link:
            # try to convert share link to direct download if contains id
            import re
            m = re.search(r'/d/([a-zA-Z0-9_-]+)', drive_link)
            if m:
                file_id = m.group(1)
                dl = f'https://drive.google.com/uc?id={file_id}&export=download'
            else:
                dl = drive_link
        else:
            dl = drive_link
        df = pd.read_csv(dl)
    except Exception as e:
        st.sidebar.error(f"Could not load drive link: {e}")

if df is None:
    st.info("Please upload a CSV or paste a public Drive link in the sidebar.")
    st.stop()

# show preview and schema
st.subheader("Data Preview")
st.dataframe(df.head(10))
st.write("Columns and types")
st.write(df.dtypes)

# --- helper functions used by pipeline ---
def ensure_revenue_column(df):
    # Normalize column names lower-cased copy
    lc = [c.lower() for c in df.columns]
    # If revenue-like exists, rename to 'revenue'
    for i, c in enumerate(lc):
        if "revenue" in c:
            df = df.rename(columns={df.columns[i]: "revenue"})
            return df
    # Try derive from price/units
    price_col = None
    qty_col = None
    for i, c in enumerate(lc):
        if "price" in c or "amount" in c and "unit" in c:
            price_col = df.columns[i]
        if "unit" in c or "quantity" in c or "qty" in c or "sold" in c:
            qty_col = df.columns[i]
    # If found both, compute
    if price_col and qty_col:
        try:
            df["revenue"] = pd.to_numeric(df[price_col], errors="coerce") * pd.to_numeric(df[qty_col], errors="coerce")
            return df
        except Exception:
            pass
    # As fallback, select the first numeric column as revenue
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df["revenue"] = df[numeric_cols[0]]
        return df
    # final fallback: zero column
    df["revenue"] = 0
    return df

def clean_code(llm_output):
    # safer cleaning: remove markdown fences and plain text lines
    code = llm_output.replace("```python","").replace("```","")
    cleaned = []
    for line in code.split("\n"):
        L = line.strip()
        if L=="":
            continue
        # remove conversational lines
        if any(phrase in L.lower() for phrase in ["here is", "sure", "i will", "below is", "the following"]):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def fix_seaborn_palette(code):
    if "sns." in code and "palette=" in code and "hue=" not in code:
        code = code.replace("palette=", "legend=False, palette_removed=")
    return code

def patch_savefig_alpha(code):
    if "plt.savefig" in code:
        code = code.replace("plt.savefig(", "plt.gcf().patch.set_facecolor('white'); plt.savefig(")
    return code

def auto_debug_code(bad_code, error):
    # use model to fix code if available
    if model is None:
        # fallback: return code unchanged with comment
        return "# Auto-debug unavailable (no API key).\n" + bad_code
    prompt = f"""The following Python code produced an error when executed.
Error:
{error}

Code:
{bad_code}

Please return only corrected Python code (no extra text)."""
    resp = model.generate_content(prompt)
    return resp.text

def ensure_print_output(code):
    # if the code doesn't have a print, print last assigned variable or df.head()
    if "print(" not in code:
        # find last assignment
        last_assign = None
        for line in reversed(code.split("\n")):
            if "=" in line and not line.strip().startswith("#"):
                last_assign = line.split("=")[0].strip()
                break
        if last_assign:
            code = code + f"\nprint({last_assign})\n"
        else:
            code = code + "\nprint(df.head())\n"
    return code

# --- Prepare dataset for analysis ---
df = ensure_revenue_column(df)  # add revenue if missing
st.write("After revenue check — column names:")
st.write(df.columns.tolist())

# Query input area
st.subheader("Ask the AI Analyst")
query = st.text_input("Enter your question (e.g. 'Show total revenue per month')", value=default_query)
run_btn = st.button("Run Analysis")

# Provide a dropdown to choose code/explain models if available
if not IMPORTED_AGENT_MODULES and model is None:
    st.warning("No agent modules imported & no API key — execution will use basic python exec. For best results, install your agent modules and set API key.")

if run_btn:
    with st.spinner("Generating analysis..."):
        # 1. Generate code using CodeGenerator if available, else create a simple default code
        if IMPORTED_AGENT_MODULES and model is not None:
            # build a small prompt for the code generator if your object needs it
            raw_code = code_gen.generate(query, df)
        else:
            # fallback simple generator — create monthly revenue aggregation
            raw_code = (
                "import pandas as pd\n"
                "df['date'] = pd.to_datetime(df.iloc[:,0])\n"
                "monthly = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum().reset_index()\n"
                "monthly['date'] = monthly['date'].astype(str)\n"
                "plt.figure(figsize=(10,4))\n"
                "plt.plot(monthly['date'], monthly['revenue'], marker='o')\n"
                "plt.title('Total Revenue Per Month')\n"
                "plt.ylabel('Total Revenue')\n"
                "plt.xticks(rotation=45)\n"
                "plt.tight_layout()\n"
                "plt.savefig('output_plot.png')\n"
                "plt.close()\n"
                "print(monthly)\n"
            )

        # show raw code (collapsible)
        st.subheader("Generated Python Code (raw)")
        st.code(raw_code, language="python")

        # 2. Clean, patch and ensure outputs
        cleaned = clean_code(raw_code)
        cleaned = fix_seaborn_palette(cleaned)
        cleaned = patch_savefig_alpha(cleaned)
        cleaned = ensure_print_output(cleaned)

        st.subheader("Cleaned & Patched Code (executing)")
        st.code(cleaned, language="python")

        # 3. Execute safely
        local_env = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns, "np": np}
        # remove old image
        if os.path.exists("output_plot.png"):
            os.remove("output_plot.png")
        exec_output = capture_output(cleaned, local_env)

        # 4. If error, attempt auto-debug once
        if exec_output and exec_output.strip().lower().startswith("error:"):
            st.warning("Execution error detected. Attempting auto-debug...")
            repaired = auto_debug_code(cleaned, exec_output)
            repaired_clean = clean_code(repaired)
            repaired_clean = patch_savefig_alpha(repaired_clean)
            repaired_clean = ensure_print_output(repaired_clean)
            st.code(repaired_clean, language="python")
            exec_output = capture_output(repaired_clean, local_env)

        st.subheader("Execution Output")
        st.text(exec_output)

        # 5. Show plot if created
        if os.path.exists("output_plot.png"):
            st.image("output_plot.png", use_column_width=True)
        else:
            st.warning("Plot not generated.")

        # 6. Explanation via ExplanationAgent or fallback
        if IMPORTED_AGENT_MODULES and model is not None:
            explanation = exp_agent.explain(query, exec_output)
        else:
            # simple fallback explanation (very basic)
            explanation = "No explanation agent available. Execution output shown above."

        st.subheader("AI Explanation")
        st.write(explanation)

        # 7. Save explanation to file + generate PDF
        os.makedirs("reports", exist_ok=True)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = f"reports/insight_{now}.txt"
        with open(txt_path, "w") as f:
            f.write(explanation)

        # ensure image is RGB (PIL re-save to strip alpha if necessary)
        if os.path.exists("output_plot.png"):
            try:
                im = Image.open("output_plot.png").convert("RGB")
                im.save("output_plot_rgb.png", format="PNG")
                image_for_pdf = "output_plot_rgb.png"
            except Exception:
                image_for_pdf = "output_plot.png"
        else:
            image_for_pdf = None

        pdf_summary = explanation if explanation else "AI Analysis Report"
        images = [image_for_pdf] if image_for_pdf else []
        pdf_path = report_gen.generate_report(title=f"AI Report: {query}", summary=pdf_summary, images=images)

        st.success(f"Report generated: {pdf_path}")
        # provide download
        with open(pdf_path, "rb") as f:
            btn = st.download_button(label="Download PDF Report", data=f, file_name=os.path.basename(pdf_path), mime="application/pdf")

st.markdown("---")
st.caption("Built with Google Gemini + Streamlit. Keep your API key safe; do not commit to source control.")
