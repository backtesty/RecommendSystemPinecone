import pinecone

pinecone.init(api_key="59395b42-e6c5-4cc2-ad29-519fca2901d9",
              environment="asia-southeast1-gcp-free")

class PineconeClient:

    def __init__(self, index_name:str, namespace:str):
        self.index_name = index_name
        self.namespace = namespace
        self.index = pinecone.Index(self.index_name)

    def upsert_vectors(self, vectors:list):
        return self.index.upsert(vectors=vectors, namespace=self.namespace)
    
    def query_similarity(self, query_vector:list, top_k=5):
        return self.index.query(
            namespace=self.namespace,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            vector=query_vector,
            filter={}
        )