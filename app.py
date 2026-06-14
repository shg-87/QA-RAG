import glob
import os

import numpy as np
from flask import Flask, request, jsonify

try:
    import faiss
except Exception:
    faiss = None

from textwave.modules.extraction.preprocessing import DocumentProcessing
from textwave.modules.extraction.embedding import Embedding
from textwave.modules.retrieval.reranker import Reranker

# TODO: Add your import statements

# TODO: You will need to implement:
# - initialize_index()
# - generate_answer()

app = Flask(__name__)

#######################################
# DEFAULT SYSTEM PARAMETERS
#######################################
STORAGE_DIRECTORY = "storage/"
CHUNKING_STRATEGY = 'fixed-length'  # or 'sentence'
CHUNKING_PARAMETERS = {
    "chunk_size": 10,
    "overlap_size": 0,
}
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_STRATEGY = "bruteforce"
INDEX_PARAMETERS = {}
# add more as needed...
RERANKING_STRATEGY = None
RERANKING_PARAMETERS = {}

INDEX = None
CHUNKS = []
EMBEDDER = None
VECTORIZER = None


def initialize_index():
    """
    1. Parse through all the documents contained in storage/corpus directory
    2. Chunk the documents using either a 'sentence' or 'fixed-length' chunking strategy.
    3. Embed each chunk using Embedding class.
    4. Store vector embeddings of these chunks in a FAISS index.
    5. Return the index.
    """
    global INDEX, CHUNKS, EMBEDDER, VECTORIZER

    if INDEX is not None:
        return INDEX

    processor = DocumentProcessing()
    corpus_directory = os.path.join(os.path.dirname(__file__), STORAGE_DIRECTORY, "corpus")
    file_paths = sorted(glob.glob(os.path.join(corpus_directory, "*.txt")) + glob.glob(os.path.join(corpus_directory, "*.clean")))

    CHUNKS = []
    for file_path in file_paths:
        if CHUNKING_STRATEGY == "sentence":
            chunks = processor.sentence_chunking(
                file_path,
                num_sentences=CHUNKING_PARAMETERS.get("num_sentences", 3),
                overlap_size=CHUNKING_PARAMETERS.get("overlap_size", 0),
            )
        else:
            chunks = processor.fixed_length_chunking(
                file_path,
                chunk_size=CHUNKING_PARAMETERS.get("chunk_size", 10),
                overlap_size=CHUNKING_PARAMETERS.get("overlap_size", 0),
            )
        CHUNKS.extend(chunks)

    if not CHUNKS:
        CHUNKS = ["No documents were found in the corpus."]

    embeddings = None
    try:
        EMBEDDER = Embedding(EMBEDDING_MODEL)
        embeddings = np.asarray(EMBEDDER.encode(CHUNKS), dtype="float32")
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        VECTORIZER = TfidfVectorizer()
        embeddings = VECTORIZER.fit_transform(CHUNKS).toarray().astype("float32")
        EMBEDDER = None

    if faiss is not None:
        INDEX = faiss.IndexFlatL2(embeddings.shape[1])
        INDEX.add(embeddings)
    else:
        INDEX = embeddings

    return INDEX


@app.route("/generate", methods=["POST"])
def generate_answer():
    """
    Generate an answer to a given query by running the retrieval and reranking pipeline.
    """
    global INDEX

    answer = None
    query = None

    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400

    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Query must be a non-empty string."}), 422

    if INDEX is None:
        initialize_index()

    if EMBEDDER is not None:
        query_embedding = np.asarray(EMBEDDER.encode([query]), dtype="float32")
    else:
        query_embedding = VECTORIZER.transform([query]).toarray().astype("float32")

    top_k = min(3, len(CHUNKS))

    if faiss is not None and hasattr(INDEX, "search"):
        _, indices = INDEX.search(query_embedding, top_k)
        context = [CHUNKS[index] for index in indices[0] if 0 <= index < len(CHUNKS)]
    else:
        distances = np.linalg.norm(INDEX - query_embedding[0], axis=1)
        indices = np.argsort(distances)[:top_k]
        context = [CHUNKS[index] for index in indices]

    # reranking
    if RERANKING_STRATEGY is not None:
        reranker = Reranker(type=RERANKING_STRATEGY)
        context, _, _ = reranker.rerank(query, context, **RERANKING_PARAMETERS)

    try:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if api_key:
            from textwave.modules.generator.question_answering import QAGeneratorMistral
            generator = QAGeneratorMistral(api_key=api_key)
            answer = generator.generate_answer(query=query, context=context)
        else:
            answer = context[0] if context else "No answer found."
    except Exception:
        answer = context[0] if context else "No answer found."

    return jsonify({"query": query, "answer": answer})



if __name__ == "__main__":
    app.run(debug=True)
