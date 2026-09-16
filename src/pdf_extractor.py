from pathlib import Path

from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (Path): The path to the PDF file.

    Returns:
        Extracted text from the PDF file as a string.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    reader = PdfReader(path)

    extracted_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            extracted_text.append(text)

    return "\n".join(extracted_text)