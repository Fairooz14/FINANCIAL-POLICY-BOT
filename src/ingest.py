
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
from typing import List
from pypdf import PdfReader
from .utils import clean_text, guess_section

@dataclass
class Chunk:
    id: str
    text: str
    page: int
    section: str

def chunk_page_text(page_text: str, page_num: int, chunk_words: int=220, overlap:int=40) -> List[Chunk]:
    words = page_text.split()
    chunks=[]; i=0; idx=0
    while i<len(words):
        start=max(0,i-(overlap if idx>0 else 0))
        end=min(len(words),i+chunk_words)
        text=" ".join(words[start:end]).strip()
        if not text: break
        section=guess_section(text)
        chunks.append(Chunk(id=f"p{page_num:03d}_c{idx:03d}",text=text,page=page_num,section=section))
        idx+=1; i+=chunk_words-overlap
        if i<=start: i=end
    return chunks

def ingest(pdf_path:str, out_path:str="index/chunks.jsonl"):
    reader=PdfReader(pdf_path)
    all_chunks=[]
    for i,page in enumerate(reader.pages,start=1):
        text=clean_text(page.extract_text() or "")
        if not text: continue
        all_chunks+=chunk_page_text(text,i)
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f:
        for ch in all_chunks: f.write(json.dumps(asdict(ch),ensure_ascii=False)+"\n")
    print(f"Wrote {len(all_chunks)} chunks to {out_path}")
