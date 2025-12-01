class ExplanationAgent:
    def __init__(self, llm):
        self.llm = llm

    def explain(self, user_query, python_output_summary):
        prompt = f"""
You are a Senior Data Analyst.

User asked: {user_query}

The analysis results were:
{python_output_summary}

Explain the findings in clear human language in 10-20 bullet points.
Avoid technical jargon.
Give some insights on revenue each month, and also provide marketing strategies for less sold product and product categories.


        """

        return self.llm(prompt)
