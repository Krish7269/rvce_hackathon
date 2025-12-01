import streamlit as st
import requests
import pandas as pd

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="AI Analyst",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    .main-title {
        font-size: 36px !important;
        font-weight: 700 !important;
        color: #2E86C1 !important;
    }
    .sub-text {
        font-size: 16px !important;
        color: #555;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #DDD;
        background-color: #FAFAFA;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload your file and enter a question to analyze data.")

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xlsx", "json"])
question = st.sidebar.text_input("Ask your question")

run_btn = st.sidebar.button("Run Analysis")

# ---------------- HEADER ----------------
st.markdown("<p class='main-title'>📊 AI Analyst</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Upload a dataset and ask a question. The AI will analyze your data and generate insights automatically.</p>", unsafe_allow_html=True)
st.write("---")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 2])

# PREVIEW SECTION
with col1:
    if uploaded_file:
        st.subheader("📄 Data Preview")
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Upload a file to preview it here.")

# RESULTS SECTION
with col2:
    if run_btn:
        if not uploaded_file:
            st.error("Please upload a file")
        elif question.strip() == "":
            st.error("Please enter a question")
        else:
            with st.spinner("Analyzing your data... Please wait."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {"question": question}
                url = "http://127.0.0.1:8000/analyze"

                try:
                    response = requests.post(url, data=data, files=files)
                    result = response.json()
                except Exception as e:
                    st.error(f"Backend error: {e}")
                    result = None

            if result:
                st.subheader("📘 Analysis Result")

                # TEXT
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.write(result.get("text", ""))
                st.markdown("</div>", unsafe_allow_html=True)

                # TABLE
                if result.get("table"):
                    st.subheader("📊 Table Output")
                    st.dataframe(pd.DataFrame(result["table"]))

                # CHART
                if result.get("chart"):
                    st.subheader("📈 Chart")
                    st.image(result["chart"])
