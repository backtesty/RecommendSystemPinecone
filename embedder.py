import requests
from pinecone_vec import PineconeClient
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def model_embedder(text):
    embedding = model.encode(text)  # np.array converted to list
    return embedding.tolist()

print("Download products...")
products = requests.get("https://fakestoreapi.com/products").json()
#print(products[0])

"""
vectors=[
        (
         "vec1",                # Vector ID 
         [0.1, 0.2, 0.3, 0.4],  # Dense vector values
         {"genre": "drama"}     # Vector metadata
        ),
        (
         "vec2", 
         [0.2, 0.3, 0.4, 0.5], 
         {"genre": "action"}
        )
    ]
"""

vectors = []
print("Embedding products...")
for product in products:
    text = """
    title: {}
    price: {}
    description: {}
    category: {}
    rating_rate: {}
    rating_count: {}
    """.format(product['title'], product['price'], product['description'], product['category'], product['rating']['rate'], product['rating']['count'])
    vector = model_embedder(text)

    vectors.append(
        (
            'vec'+str(product['id']),
            vector,
            {
                'title': product['title'],
                'price': product['price'],
                'description': product['description'],
                'category': product['category'],
                'rating_rate': product['rating']['rate'],
                'rating_count': product['rating']['count'],
                'image': product['image'],
            }
        )
    )


index_name = "ianewsindex"
namespace = "ianews"
print("Creating Pinecone vectors cloud...")
p_client = PineconeClient(index_name, namespace)
p_client.upsert_vectors(vectors)
print("Done")
    
