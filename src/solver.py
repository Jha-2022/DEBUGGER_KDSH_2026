

import pandas as pd
from indexer import build_index
from retriever import retrieve_evidence
from verifier import decompose_backstory, analyze_claim

def solve(csv_path, novel_dir, output_path):
    df = pd.read_csv(csv_path)
    index = build_index(novel_dir)

    outputs = []

    for _, row in df.iterrows():
        claims = decompose_backstory(row["backstory"])
        analyses = []

        for claim in claims:
            evidence = retrieve_evidence(index, claim)
            analyses.append(analyze_claim(claim, evidence))

        final_label = int(all(a["verdict"] == 1 for a in analyses))

        outputs.append({
            "id": row["id"],
            "prediction": final_label,
            "analysis": analyses
        })

    pd.DataFrame(outputs).to_csv(output_path, index=False)

