import pathway as pw
from pathway.stdlib.ml.embedding import SentenceTransformerEmbedding

def build_index(novel_dir: str):
    """
    Builds a vector index over full novels using Pathway-native embeddings.
    """

    files = pw.io.fs.read(
        novel_dir,
        format="text",
        with_metadata=True
    )

    # Chunk long narrative
    chunks = files.select(
        text=pw.this.data,
        metadata=pw.this.metadata
    ).flat_map(
        lambda row: pw.stdlib.text.splitter(
            row.text,
            chunk_size=800,
            overlap=200
        )
    )

    embedder = SentenceTransformerEmbedding(
        model="all-MiniLM-L6-v2"
    )

    vectors = chunks.select(
        vector=embedder(chunks.text),
        text=chunks.text,
        metadata=chunks.metadata
    )

    index = pw.stdlib.ml.index.KNNIndex(
        vectors,
        n_dimensions=384
    )

    return index
