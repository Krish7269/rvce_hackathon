from textwrap import dedent

class Reporter:
    def generate_text_report(self, df, cleaned_info, forecast_info=None):
        report = ""

        report += "# DATA ANALYSIS REPORT\n\n"

        # Summary
        report += "## Summary Statistics\n"
        report += df.describe().to_string()
        report += "\n\n"

        # Missing values
        report += "## Data Cleaning Insights\n"
        report += f"- Missing Values: {cleaned_info['missing_values']}\n"
        report += f"- Duplicates Removed: {cleaned_info['duplicates_removed']}\n\n"

        # Forecast
        if forecast_info:
            report += "## Forecast (Next Values)\n"
            report += f"{forecast_info}\n\n"

        # Recommendations
        report += "## Recommendations\n"
        report += dedent("""
            - Monitor upward/downward trends closely.
            - Consider deeper segmentation for better insights.
            - Improve data collection for missing entries.
            - Use more advanced forecasting models for accuracy.
        """)

        return report
