from PyPDF2 import PdfReader

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text_content = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text_content += page.extract_text() + "\n"
        return text_content   # return plain string, not dict
    else:
        raise FileNotFoundError("No file uploaded")
