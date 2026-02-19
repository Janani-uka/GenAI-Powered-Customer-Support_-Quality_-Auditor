from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid

model = SentenceTransformer('all-MiniLM-L6-v2')

pc = Pinecone(api_key="pcsk_2EyRxK_2N8evfGxxUjx4TfMRw4ixZyJBbSDgiTAEyX5RVxPpNDvCMWdTRwkm1dkWaWyhgW")
index = pc.Index("policy-index")

# Multiple policy chunks
policies = [
"""
Refund Policy:
Customers can request refund within 30 days of purchase.
""",

"""
Empathy Policy:
Agent must acknowledge customer issue and apologize sincerely.
""",

"""
Greeting Policy:
Agent must greet customer politely at the start of conversation.
""",

"""
Escalation Policy:
If issue cannot be resolved, agent must escalate to supervisor.
""",

"""
Professional Language Policy:
Agent must avoid rude or abusive words and maintain professionalism.
""",

"""
Closing Policy:
Agent must thank customer and close conversation politely.
"""
]

for policy in policies:
    vector = model.encode(policy).tolist()

    index.upsert([
        {
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {"text": policy}
        }
    ])

print("✅ All policies inserted!")
