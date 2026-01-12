import re

def decompose_backstory(backstory: str):
    """
    Breaks backstory into atomic claims using rule-based NLP.
    """
    sentences = re.split(r"[.?!]", backstory)
    claims = [s.strip() for s in sentences if len(s.strip()) > 15]
    return claims


def analyze_claim(claim, evidence):
    """
    Determines whether a claim is:
    - Supported
    - Contradicted
    - Weakly supported (constraint violation)
    """

    support_hits = 0

    for e in evidence:
        overlap = set(claim.lower().split()) & set(e["excerpt"].lower().split())
        if len(overlap) > 3:
            support_hits += 1

    if support_hits >= 2:
        verdict = 1
        status = "Supported"
    elif support_hits == 1:
        verdict = 0
        status = "Narrative Constraint Violation"
    else:
        verdict = 0
        status = "Contradicted"

    return {
        "claim": claim,
        "verdict": verdict,
        "status": status,
        "evidence": evidence
    }
