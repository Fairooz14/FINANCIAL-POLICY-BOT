import os, json, pickle
import numpy as np

INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
EMB_META_PATH = os.path.join(INDEX_DIR, "emb_meta.pkl")
TFIDF_VEC_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")
TFIDF_MAT_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.pkl")

def _load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

class Retriever:
    def __init__(self):
        self.chunks = _load_chunks()
        self.backend = None
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.faiss = faiss
            if os.path.exists(FAISS_PATH):
                self.index = faiss.read_index(FAISS_PATH)
                with open(EMB_META_PATH, "rb") as f: self.emb_meta = pickle.load(f)
            else:
                texts = [c["text"] for c in self.chunks]
                X = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                self.index = faiss.IndexFlatIP(X.shape[1]); self.index.add(X.astype("float32"))
                with open(EMB_META_PATH, "wb") as f: pickle.dump({"ids":[c["id"] for c in self.chunks]}, f)
                faiss.write_index(self.index, FAISS_PATH)
            self.backend = "faiss"
        except Exception as e:
            print("Falling back to TF-IDF:", e)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.cosine_similarity = cosine_similarity
            self.backend = "tfidf"
            if os.path.exists(TFIDF_VEC_PATH):
                with open(TFIDF_VEC_PATH,"rb") as f: self.vectorizer = pickle.load(f)
                with open(TFIDF_MAT_PATH,"rb") as f: self.tfidf_matrix = pickle.load(f)
            else:
                texts = [c["text"] for c in self.chunks]
                self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
                with open(TFIDF_VEC_PATH,"wb") as f: pickle.dump(self.vectorizer,f)
                with open(TFIDF_MAT_PATH,"wb") as f: pickle.dump(self.tfidf_matrix,f)

    def search(self, query: str, top_k: int = 4):
        if self.backend == "faiss":
            qv = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            D,I = self.index.search(qv.astype("float32"), top_k)
            return [(float(D[0][i]), self.chunks[I[0][i]]) for i in range(min(top_k,len(self.chunks)))]
        else:
            qv = self.vectorizer.transform([query])
            sims = self.cosine_similarity(qv, self.tfidf_matrix)[0]
            idxs = np.argsort(-sims)[:top_k]
            return [(float(sims[i]), self.chunks[i]) for i in idxs]
