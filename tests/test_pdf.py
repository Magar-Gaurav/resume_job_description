from resume_job_description.src.pdf_extractor import extract_text_from_pdf

pdf_path = "data/raw/resume.pdf"  

text = extract_text_from_pdf(pdf_path)

print("\n==== Extracted Text from PDF ====\n")
print(text)