# search documents
from pinecone_vec import PineconeClient
from embedder import model_embedder

text_search = "Mens Casual T-Shirts"
vector_search = model_embedder(text_search)

index_name = "ianewsindex"
namespace = "ianews"
p_client = PineconeClient(index_name, namespace)
result = p_client.query_similarity(vector_search)
print(result)

