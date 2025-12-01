import json

class CodeGenerator:
    def __init__(self, llm):
        self.llm = llm

    def get_column_info(self, df):
        return {col: str(df[col].dtype) for col in df.columns}

    def generate(self, user_query, df):
        schema = self.get_column_info(df)

        prompt = f"""
You are a Python data analyst.
User query: "{user_query}"

Dataset columns and types:
{json.dumps(schema, indent=2)}

Write Python code using pandas/matplotlib/seaborn to answer the question.
DO NOT write explanations. ONLY Python code.
Assume the dataframe is available as df.
Plots must display automatically.
            """

        response = self.llm(prompt)
        return response
