import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import ollama
import re

# ---------- LOAD MODEL ----------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- CONNECT PINECONE ----------
pc = Pinecone(api_key="pcsk_2EyRxK_2N8evfGxxUjx4TfMRw4ixZyJBbSDgiTAEyX5RVxPpNDvCMWdTRwkm1dkWaWyhgW")
index = pc.Index("support-policy-index")

# ---------- QUERY GENERATION ----------
def generate_audit_query(transcript):
    query = f"""
Check whether the following support agent response follows customer support policies
like empathy, greeting, professionalism and compliance.

Response:
{transcript}
"""
    return query

# ---------- RETRIEVE POLICY FROM PINECONE ----------
def retrieve_policy(text):
    audit_query = generate_audit_query(text)
    vector = model.encode(audit_query).tolist()

    result = index.query(
        vector=vector,
        top_k=1,
        include_metadata=True
    )

    return result['matches'][0]['metadata']['text']

# ---------- LLM AUDIT + SCORE EXTRACTION ----------
def llm_audit(transcript, policy):
    prompt = f"""
You are a support quality auditor.

Transcript:
{transcript}

Policy:
{policy}

Give scores out of 10 in this exact format:

Empathy: <score>
Professionalism: <score>
Compliance: <score>

Then explain violations and suggestions.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    output = response['message']['content']

    # ---- Extract scores ----
    empathy = re.search(r"Empathy:\s*(\d+)", output)
    prof = re.search(r"Professionalism:\s*(\d+)", output)
    comp = re.search(r"Compliance:\s*(\d+)", output)

    empathy_score = int(empathy.group(1)) if empathy else 0
    prof_score = int(prof.group(1)) if prof else 0
    comp_score = int(comp.group(1)) if comp else 0

    final_score = round((empathy_score + prof_score + comp_score) / 30 * 100, 2)

    return output, empathy_score, prof_score, comp_score, final_score


# ---------- MAIN PROCESS ----------
df = pd.read_csv("cleaned_transcripts.csv")

policy_contexts = []
llm_outputs = []
empathy_list = []
prof_list = []
comp_list = []
final_scores = []

for text in df["Clean_Text"]:
    print("Auditing:", text[:50])

    policy = retrieve_policy(text)
    audit_text, e, p, c, final = llm_audit(text, policy)

    policy_contexts.append(policy)
    llm_outputs.append(audit_text)
    empathy_list.append(e)
    prof_list.append(p)
    comp_list.append(c)
    final_scores.append(final)

df["Policy_Context"] = policy_contexts
df["LLM_Audit"] = llm_outputs
df["Empathy"] = empathy_list
df["Professionalism"] = prof_list
df["Compliance"] = comp_list
df["Final_Score_Out_of_100"] = final_scores

df.to_csv("audited_output.csv", index=False)

print("✅ Auditing completed. File saved as audited_output.csv")
