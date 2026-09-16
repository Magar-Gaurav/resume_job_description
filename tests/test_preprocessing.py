from resume_job_description.src.pdf_extractor import extract_text_from_pdf
from resume_job_description.src.preprocessing import preprocess_text

pdf_path = "data/raw/resume.pdf"

# Extract raw resume text from the PDF
raw_text = extract_text_from_pdf(pdf_path)

# Preprocess the extracted text
preprocessed_text = preprocess_text(raw_text)

print("\n==== Raw Text from PDF ====\n")
print(raw_text)

print("\n==== Preprocessed Text from PDF ====\n")
print(preprocessed_text)