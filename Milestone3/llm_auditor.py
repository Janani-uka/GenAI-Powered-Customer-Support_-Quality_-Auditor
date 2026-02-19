import requests

def llm_score(transcript, policy):
    prompt = f"""
You are a support quality auditor.

Policy:
{policy}

Transcript:
{transcript}

Answer clearly:
1. Is the agent compliant? (Yes/No)
2. Score out of 10
3. Violation explanation
4. Suggestion to improve
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]
