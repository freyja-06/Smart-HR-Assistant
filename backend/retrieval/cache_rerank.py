from functools import lru_cache
import hashlib

# create cache global
def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# cache wrapper
rerank_cache = {}

def get_cached_score(query, doc):
    key = (_hash(query), _hash(doc))

    return rerank_cache.get(key)

def set_cached_score(query, doc, score):
    key = (_hash(query), _hash(doc))
    rerank_cache[key] = score