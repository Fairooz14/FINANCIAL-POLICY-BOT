import argparse
from .ingest import ingest
from .retrieval import Retriever

def build(pdf_path: str):
    ingest(pdf_path, "index/chunks.jsonl")
    _ = Retriever()
    print("Index built.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to policy PDF")
    args = ap.parse_args()
    build(args.pdf)
