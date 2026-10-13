import chromadb
from sentence_transformers import SentenceTransformer

from .chunking import chunk_doc, load_docs
from .fetch import RAW_DIR, ROOT

CHROMA_DIR = ROOT / ".chroma"

# e5 models need "query: " / "passage: " prefixes; the others do not
MODELS = {
    "minilm": {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "q": "", "p": ""},
    "e5-small": {"name": "intfloat/multilingual-e5-small", "q": "query: ", "p": "passage: "},
    "bge-m3": {"name": "BAAI/bge-m3", "q": "", "p": ""},
}


class Embedder:
    def __init__(self, key: str, use_prefix: bool = True):
        cfg = MODELS[key]
        self.model = SentenceTransformer(cfg["name"])
        self.model.max_seq_length = min(self.model.max_seq_length, 512)
        self.q = cfg["q"] if use_prefix else ""
        self.p = cfg["p"] if use_prefix else ""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode([self.p + t for t in texts], normalize_embeddings=True,
                                 show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.q + text], normalize_embeddings=True)[0].tolist()


def build_index(model_key: str, max_chars: int = 800, with_header: bool = True, use_prefix: bool = True):
    embedder = Embedder(model_key, use_prefix)
    chunks = [c for d in load_docs(RAW_DIR) for c in chunk_doc(d, max_chars, with_header)]
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = f"{model_key}_{max_chars}_{'h' if with_header else 'n'}_{'p' if use_prefix else 'x'}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
            embeddings=embedder.embed_passages([c["text"] for c in chunks]),
            metadatas=[c["meta"] for c in chunks])
    return col, embedder, len(chunks)


def search(col, embedder: Embedder, query: str, k: int = 5) -> list[dict]:
    res = col.query(query_embeddings=[embedder.embed_query(query)], n_results=k,
                    include=["documents", "metadatas", "distances"])
    return [{"text": d, "meta": m, "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]