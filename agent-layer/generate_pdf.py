import os
import subprocess
import markdown_it

md_path = r"c:\Users\abhay\Desktop\Projects\AI-Assisted Fault Diagnosis and Technical Information Retrieval for Electric Two-Wheelers\ev-fault-diagnosis-rag\docs\fine_tuning_technical_report.md"
html_path = r"c:\Users\abhay\Desktop\Projects\AI-Assisted Fault Diagnosis and Technical Information Retrieval for Electric Two-Wheelers\ev-fault-diagnosis-rag\docs\fine_tuning_technical_report.html"
pdf_path = r"c:\Users\abhay\Desktop\Projects\AI-Assisted Fault Diagnosis and Technical Information Retrieval for Electric Two-Wheelers\ev-fault-diagnosis-rag\docs\fine_tuning_technical_report.pdf"

with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Convert Markdown to HTML
md = markdown_it.MarkdownIt("commonmark", {"breaks": True, "html": True}).enable("table")
body_html = md.render(md_content)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Domain-Adapted Dense Retrieval for EV Fault Diagnosis - Technical Report</title>
<style>
  @page {{
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1a202c;
    line-height: 1.55;
    font-size: 11pt;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    font-size: 20pt;
    color: #1a365d;
    margin-bottom: 4px;
    border-bottom: 2px solid #2b6cb0;
    padding-bottom: 8px;
    line-height: 1.25;
  }}
  h2 {{
    font-size: 14pt;
    color: #2b6cb0;
    margin-top: 20px;
    margin-bottom: 8px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
  }}
  h3 {{
    font-size: 12pt;
    color: #2d3748;
    margin-top: 14px;
    margin-bottom: 6px;
  }}
  p, li {{
    color: #2d3748;
    margin-bottom: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 9.5pt;
  }}
  th, td {{
    border: 1px solid #cbd5e0;
    padding: 7px 10px;
    text-align: left;
  }}
  th {{
    background-color: #ebf8ff;
    color: #2b6cb0;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: #f7fafc;
  }}
  strong {{
    color: #1a202c;
  }}
  code {{
    font-family: Consolas, Monaco, "Courier New", monospace;
    background-color: #edf2f7;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 9.5pt;
    color: #805ad5;
  }}
  pre {{
    background-color: #2d3748;
    color: #f7fafc;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 9pt;
  }}
  blockquote {{
    border-left: 4px solid #3182ce;
    margin: 12px 0;
    padding: 8px 16px;
    background-color: #ebf8ff;
    color: #2c5282;
  }}
  hr {{
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 20px 0;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    background-color: #c6f6d5;
    color: #22543d;
    border-radius: 4px;
    font-weight: bold;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"HTML generated at: {html_path}")

# Run Edge headless to print to PDF
edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
cmd = [
    edge_exe,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    html_path
]

print("Rendering PDF with Microsoft Edge headless...")
result = subprocess.run(cmd, capture_output=True, text=True)
if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
    print(f"PDF successfully generated: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
else:
    print("PDF generation failed:", result.stderr)
