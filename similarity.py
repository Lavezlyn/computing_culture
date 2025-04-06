import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("/aifs4su/yaodong/sitong/projects/nlp/ru_keywords.json", "r") as f:
    ru_keywords = json.load(f)
    
with open("/aifs4su/yaodong/sitong/projects/nlp/fa_keywords.json", "r") as f:
    fa_keywords = json.load(f)

def is_similar(ru_keyword, fa_keyword):
    for char in ru_keyword:
        if char in fa_keyword:
            return True
    return False

def normalize_similarity(sim):
    """Normalize similarity score to [0,1] range"""
    return (sim + 1) / 2

results = []

for ru_keyword in ru_keywords:
    for fa_keyword in fa_keywords:
        if is_similar(ru_keyword["keyword"], fa_keyword["keyword"]):
            # Reshape the embeddings to 2D arrays before calculating cosine similarity
            ru_embedding = np.array(ru_keyword["embedding"]).reshape(1, -1)
            fa_embedding = np.array(fa_keyword["embedding"]).reshape(1, -1)
            
            # Calculate raw cosine similarity
            raw_cosine = cosine_similarity(ru_embedding, fa_embedding)[0][0]
            
            # Calculate normalized cosine similarity (mapped to [0,1])
            norm_cosine = normalize_similarity(raw_cosine)
            
            # Calculate euclidean similarity (always positive)
            euclidean_dist = np.linalg.norm(ru_embedding - fa_embedding)
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # Calculate manhattan similarity (always positive)
            manhattan_dist = np.sum(np.abs(ru_embedding - fa_embedding))
            manhattan_sim = 1 / (1 + manhattan_dist)
            
            
            results.append({
                "ru_keyword": {
                    "keyword": ru_keyword["keyword"],
                    "source": ru_keyword["source"],
                    "chapter": ru_keyword["chapter"],
                    "pos": ru_keyword["pos"],
                    "score": ru_keyword["score"],
                    "synonym": ru_keyword["similarities"]
                },
                "fa_keyword": {
                    "keyword": fa_keyword["keyword"],
                    "source": fa_keyword["source"],
                    "chapter": fa_keyword["chapter"],
                    "pos": fa_keyword["pos"],
                    "score": fa_keyword["score"],
                    "synonym": fa_keyword["similarities"]
                },
                "similarities": {
                    "cosine": float(norm_cosine),
                    "euclidean": float(euclidean_sim),
                    "manhattan": float(manhattan_sim),
                }
            })

with open("similarity_selected.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

    
    
    
    