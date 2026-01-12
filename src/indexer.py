import pathway as pw
from pathway.stdlib.ml.index import KNNIndex
from sentence_transformers import SentenceTransformer

# 1. Configuration
DATA_DIR = "./data/"  # Folder where you keep "In search of the castaways.txt", etc.
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Fast, efficient local model

# 2. Define the Embedding UDF (User Defined Function)
# We wrap the model loading to ensure it runs efficiently in the Pathway dataflow
class Embedder(pw.UDF):
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def __call__(self, text):
        return self.model.encode(text).tolist()

# 3. The Pathway Pipeline
def run_indexing_server():
    # A. Read the files from the folder
    # mode="static" because the hackathon files are fixed, not a live stream
    files = pw.io.fs.read(
        DATA_DIR,
        format="binary",
        mode="static",
        with_metadata=True
    )

    # B. Decode and Clean
    # We convert binary to utf-8
    documents = files.select(
        text=pw.this.data.decode("utf-8"),
        filename=pw.this._metadata.path  # We need this to match the book to the backstory
    )

    # C. Chunking
    # Splitting 100k+ words into manageable 500-character chunks with overlap
    # This preserves narrative context better than strict line splitting
    chunks = documents.select(
        filename=pw.this.filename,
        chunk=pw.udfs.split_text(pw.this.text, sep=" ", max_size=500) # Simple semantic splitting
    ).flatten(pw.this.chunk)

    # D. Embedding
    # Apply the embedding model to each chunk
    embedder = Embedder()
    embeddings = chunks.select(
        filename=pw.this.filename,
        text=pw.this.chunk,
        vector=embedder(pw.this.chunk)
    )

    # E. Build the KNN Index
    # This creates the searchable structure
    index = KNNIndex(
        embeddings.vector,
        embeddings,
        n_dimensions=384 # Dimension of MiniLM-L6-v2
    )

    # F. Expose as a Server
    # This allows our Solver script to query it like a database
    pw.io.http.write_groups(
        index.query,
        host="127.0.0.1",
        port=8000,
        name="narrative_search"
    )

    pw.run()

if __name__ == "__main__":
    run_indexing_server()