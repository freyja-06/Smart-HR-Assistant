from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

directory_path: str = BASE_DIR / "data" / "All_pdf_cv"

print(directory_path)


