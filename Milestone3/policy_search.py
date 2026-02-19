# policy_search.py

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# -----------------------------
# 1. Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 2. Connect to Pinecone
# -----------------------------
pc = Pinecone(api_key="pcsk_2EyRxK_2N8evfGxxUjx4TfMRw4ixZyJBbSDgiTAEyX5RVxPpNDvCMWdTRwkm1dkWaWyhgW")   # same key
index = pc.Index("newproject")    # same 384 index

# -----------------------------
# 3. Ask user question
# -----------------------------
query = input("Enter your question: ")

# Convert question to embedding
query_embedding = model.encode(query).tolist()

# -----------------------------
# 4. Search in Pinecone
# -----------------------------
results = index.query(
    vector=query_embedding,
    top_k=1,
    include_metadata=True
)

# -----------------------------
# 5. Show matched policy text
# -----------------------------
match = results["matches"][0]["metadata"]["text"]

print("\n🔍 Most relevant policy found:\n")
print(match)
