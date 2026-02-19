from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

model = SentenceTransformer('all-MiniLM-L6-v2')

pc = Pinecone(api_key="pcsk_2EyRxK_2N8evfGxxUjx4TfMRw4ixZyJBbSDgiTAEyX5RVxPpNDvCMWdTRwkm1dkWaWyhgW")
index = pc.Index("policy-index")  # your 384-dim index


def evaluate_transcript(text):
    query_vector = model.encode(text).tolist()

    result = index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True
    )

    policy_context = result['matches'][0]['metadata']['text']
    return policy_context
