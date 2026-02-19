from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_2EyRxK_2N8evfGxxUjx4TfMRw4ixZyJBbSDgiTAEyX5RVxPpNDvCMWdTRwkm1dkWaWyhgW")
index = pc.Index("policy-index")

index.delete(delete_all=True)
print("✅ Index cleared")
