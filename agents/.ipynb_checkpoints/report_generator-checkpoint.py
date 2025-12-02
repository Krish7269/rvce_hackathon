from fpdf import FPDF
import datetime
import os

class ReportGenerator:

    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def add_title(self, title):
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 18)
        self.pdf.cell(0, 10, title, ln=True, align="C")
        self.pdf.ln(5)

    def add_summary(self, summary_text):
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 8, summary_text)
        self.pdf.ln(5)

    def add_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                self.pdf.set_font("Arial", "I", 12)
                self.pdf.multi_cell(0, 8, f"[Image Missing: {image_path}]")
                self.pdf.ln(5)
                return

            if os.path.getsize(image_path) < 1024:  # less than 1 KB
                self.pdf.set_font("Arial", "I", 12)
                self.pdf.multi_cell(0, 8, f"[Image Invalid or Empty: {image_path}]")
                self.pdf.ln(5)
                return

            self.pdf.image(image_path, w=170)
            self.pdf.ln(5)

        except Exception as e:
            self.pdf.set_font("Arial", "I", 12)
            self.pdf.multi_cell(0, 8, f"[Error Loading Image: {str(e)}]")
            self.pdf.ln(5)

    def save(self, filename="analysis_report.pdf"):
        self.pdf.output(filename)
        return filename

    def generate_report(self, title, summary, images=[]):
        self.add_title(title)
        self.add_summary(summary)

        for img in images:
            self.add_image(img)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.pdf"
        return self.save(filename)
