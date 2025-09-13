from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text

noli_text = extract_text_from_pdf("./noypi-noli-me-tangere-buod-ng-bawat-kabanata-1-64-with-talasalitaan.pdf")
elfili_text = extract_text_from_pdf("./pinoycollection.com-el-filibusterismo-buod-ng-bawat-kabanata-1-39-with-talasalitaan.pdf")

full_text = noli_text + " " + elfili_text
