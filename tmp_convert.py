import sys
from docling.document_converter import DocumentConverter

try:
    converter = DocumentConverter()
    result = converter.convert("docs/thesis-paper/latest-paper.pdf")
    md_content = result.document.export_to_markdown()
    with open("docs/thesis-paper/latest-paper.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Conversion successful.")
except Exception as e:
    print(f"Error: {e}")
