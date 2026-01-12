import pandas as pd
import requests
import json
import time
import google.generativeai as genai
from google.api_core import retry

# --- Configuration ---
PATHWAY_HOST = "http://127.0.0.1:8000/v1/narrative_search"

# Replace with your actual key
GOOGLE_API_KEY = "AIzaSyCBJGczXLoPnWDKAbABAG6kFE18livnIXg" 

# Configure the SDK
genai.configure(api_key=GOOGLE_API_KEY)

# Select Models
# 'gemini-1.5-flash' is fast and cheap (good for splitting text)
# 'gemini-1.5-pro' is better at reasoning (good for the final judge)
DECOMPOSER_MODEL = genai.GenerativeModel('gemini-1.5-flash')
JUDGE_MODEL = genai.GenerativeModel('gemini-1.5-pro')

def clean_json_text(text):
    """
    Helper to strip markdown formatting (```json ... ```) from LLM responses
    to prevent JSON decoding errors.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def query_pathway(query_text, filename, k=5):
    """
    Queries the Pathway server for the most relevant chunks.
    """
    try:
        response = requests.post(
            PATHWAY_HOST,
            json={
                "query": query_text,
                "k": k,
                # Note: Ensure your Pathway indexer supports this filter syntax.
                # If not, fetch k=20 and filter inside this python script.
                "filter": f"filename == '{filename}'" 
            },
            timeout=10
        )
        if response.status_code == 200:
            # Pathway usually returns a list of objects or a specific dictionary structure
            # Adjust 'data' vs direct list based on your specific indexer output
            return response.json() 
    except Exception as e:
        print(f"Error querying Pathway: {e}")
    return []

def decompose_backstory(backstory):
    """
    Uses Gemini Flash to break the backstory into atomic claims.
    """
    prompt = f"""
    You are a data processing assistant. Break the following character backstory into 3-5 atomic, verifiable facts.
    Ignore vague feelings; focus on concrete events, relationships, locations, or physical traits.

    Backstory: "{backstory}"

    Return ONLY a valid JSON list of strings. Example: ["He was born in Paris", "He hates dogs"]
    """
    
    try:
        response = DECOMPOSER_MODEL.generate_content(prompt)
        cleaned_text = clean_json_text(response.text)
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"Error decomposing backstory: {e}")
        # Fallback: treat the whole text as one claim
        return [backstory] 

def verify_consistency(claims, book_filename):
    """
    Checks claims against the novel text using Gemini Pro.
    """
    evidence_log = []
    is_consistent = 1
    
    for claim in claims:
        # 1. Retrieve Evidence
        relevant_chunks = query_pathway(claim, book_filename)
        
        if not relevant_chunks:
            continue # No evidence found, assume consistent (benefit of doubt)

        # Extract text from Pathway response (adjust key 'text' if your indexer uses 'chunk')
        # Assuming list of dicts: [{'text': '...'}, {'text': '...'}]
        context_block = "\n---\n".join([r.get('text', '') for r in relevant_chunks])
        
        # 2. Gemini Judge
        prompt = f"""
        You are an expert literary consistency checker.
        
        CLAIM: "{claim}"
        
        EVIDENCE FROM NOVEL:
        {context_block}
        
        TASK:
        Determine if the CLAIM explicitly contradicts the EVIDENCE.
        - If the text explicitly refutes the claim (e.g., Claim="He is poor", Text="He bought a golden castle"), label is 0.
        - If the text supports the claim OR is unrelated/silent, label is 1.
        
        OUTPUT FORMAT:
        Return ONLY valid JSON:
        {{
            "label": 0 or 1,
            "rationale": "One short sentence explaining why.",
            "excerpt": "Verbatim quote from evidence if contradiction found, else empty string."
        }}
        """
        
        try:
            response = JUDGE_MODEL.generate_content(prompt)
            result = json.loads(clean_json_text(response.text))
            
            # Logic: If ANY claim is a contradiction (0), the whole story is inconsistent.
            if result.get('label') == 0:
                is_consistent = 0
                evidence_log.append(result)
                break # Stop checking other claims, we found a lie.
            
        except Exception as e:
            print(f"Error verifying claim '{claim}': {e}")
            
    return is_consistent, evidence_log

def main():
    # Load Data
    # Ensure columns match your actual CSV headers
    try:
        df_test = pd.read_csv("data/test.csv") # Updated path to likely location
    except FileNotFoundError:
        df_test = pd.read_csv("test.csv")

    results = []
    
    print(f"Processing {len(df_test)} rows with Gemini...")
    
    for index, row in df_test.iterrows():
        print(f"Analyzing Story {index + 1}/{len(df_test)}...")
        
        story_id = row.get('story_id')
        backstory = row.get('backstory')
        filename = row.get('story_file_name')
        
        if not all([story_id, backstory, filename]):
            print(f"Skipping row {index} due to missing data.")
            continue

        # 1. Decompose
        claims = decompose_backstory(backstory)
        
        # 2. Verify
        prediction, evidence = verify_consistency(claims, filename)
        
        # 3. Format Output
        if prediction == 0 and evidence:
            rationale_text = f"{evidence[0]['rationale']} (Excerpt: '{evidence[0]['excerpt']}')"
        else:
            rationale_text = "No direct contradictions found in retrieved narrative segments."
        
        results.append({
            "Story ID": story_id,
            "Prediction": prediction,
            "Rationale": rationale_text
        })
        
        # Optional: Sleep to respect rate limits if using free tier
        time.sleep(1) 
        
    # Save Results
    output_file = "outputs/results.csv"
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    main()