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
    
    MANDATORY RULES:
    1. Assume dataframe is available as df.
    2. If you generate a plot, ALWAYS end with:
       plt.savefig('output_plot.png')
    3. NEVER show plt.show()
    4. NEVER print huge dataframes.
    5. Only output Python code. No explanations.
    6. Always create at least one matplotlib plot.
    7. Always end the script with:
       plt.savefig('output_plot.png')
    8. Do NOT call plt.show().

    Use ONLY matplotlib for plotting. Avoid seaborn.
    When saving a plot:
    - ALWAYS use: plt.savefig("output_plot.png", format="png", dpi=200, bbox_inches="tight", facecolor="white")
    - ALWAYS call: plt.close()


    AFTER performing any analysis:
- ALWAYS print the computed values using `print()`
- Examples:
    print(monthly_revenue)
    print(result_df)
    print(total_sales)

NEVER return empty output. ALWAYS print something meaningful.


If the dataset does not contain a 'revenue' column, 
automatically compute it using:

revenue = Unit_Price * Quantity

If such columns are missing, choose the most relevant 
numeric column as revenue.

    
    Now write the code:
            """

        response = self.llm(prompt)
        return response

