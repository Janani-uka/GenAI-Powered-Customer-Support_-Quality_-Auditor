from pinecone import Pinecone

PINECONE_API_KEY = "pcsk_2EyRxK_2N8evfGxxUjx4TfMRw4ixZyJBbSDgiTAEyX5RVxPpNDvCMWdTRwkm1dkWaWyhgW"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host="https://policy-index-h3oqus5.svc.aped-4627-b74a.pinecone.io")

query = "agent did not verify customer identity"

results = index.search(
    namespace="policy",
    query={
        "top_k": 3,
        "inputs": {"text": query}
    }
)

print(results)
