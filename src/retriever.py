def retrieve_evidence(index, query: str, k: int = 8):
    """
    Retrieves top-k semantically relevant excerpts.
    """

    results = index.search(query, k=k)

    return [
        {
            "excerpt": r.text,
            "metadata": r.metadata
        }
        for r in results
    ]
