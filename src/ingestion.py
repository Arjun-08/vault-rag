from pathlib import Path
import re
from pypdf import PdfReader
from docx import Document

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def read_pdf(path):
    reader = PdfReader(str(path)); records=[]
    for page_num,page in enumerate(reader.pages,1):
        text=clean_text(page.extract_text() or "")
        if text: records.append({"document_id":path.stem,"document_name":path.name,"page":page_num,"text":text})
    return records

def read_docx(path):
    doc=Document(str(path)); text=clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    return [{"document_id":path.stem,"document_name":path.name,"page":None,"text":text}] if text else []

def read_text(path):
    text=clean_text(path.read_text(encoding="utf-8",errors="ignore"))
    return [{"document_id":path.stem,"document_name":path.name,"page":None,"text":text}] if text else []

def load_file(path):
    s=path.suffix.lower()
    if s==".pdf": return read_pdf(path)
    if s==".docx": return read_docx(path)
    if s in {".txt",".md"}: return read_text(path)
    return []

def load_directory(directory):
    paths=sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf",".docx",".txt",".md"})
    records=[]
    for p in paths:
        print(f"[INGEST] Reading {p.name}"); records.extend(load_file(p))
    print(f"[INGEST] Files: {len(paths)} | Pages/records: {len(records)}")
    return records
