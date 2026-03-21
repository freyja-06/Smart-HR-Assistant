from sklearn.metrics.pairwise import cosine_similarity
from backend.agents.llm_processor.llm_factory import ModelFactory
import numpy as np
from typing import List
from backend.config.database import *
from rank_bm25 import BM25Okapi
import backend.constant_variables as const

def build_history_store(
    docs: List,
    embedding_model,
    max_docs: int = 100
):
    """
    Xây dựng history store từ List[Document]
    """

    # 1. Giới hạn số lượng docs
    docs = docs[:max_docs]

    # 2. Tạo corpus cho BM25
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [text.split() for text in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    # 3. Embed docs
    embeddings = embedding_model.embed_documents(corpus)

    # 4. Lưu lại
    history_store = {
        "docs": docs,
        "embeddings": embeddings,
        "bm25": bm25,
        "corpus": corpus
    }

    return history_store

def update_history_store(
    history_store,
    new_docs: List,
    embedding_model,
    max_docs: int = 100
):
    """
    Update history store với docs mới
    """

    # 1. Merge docs
    all_docs = history_store["docs"] + new_docs

    # 2. Remove duplicate (quan trọng!)
    seen = set()
    unique_docs = []

    for doc in all_docs:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    # 3. Giới hạn size (sliding window)
    unique_docs = unique_docs[-max_docs:]

    # 4. Rebuild store
    return build_history_store(
        docs=unique_docs,
        embedding_model=embedding_model,
        max_docs=max_docs
    )


def fusion_retrieval(
    vectorstore,
    all_docs,
    doc_embeddings,
    bm25,
    query: str,
    k: int = 50,
    alpha: float = 0.5,
    top_k_vector: int = 50,
    top_k_bm25: int = 50,
):
    epsilon = 1e-8

    # --- Vector search (top-k)
    vector_results = vectorstore.similarity_search_with_score(query, k=top_k_vector)
    
    vector_docs = [doc for doc, _ in vector_results]
    vector_scores_raw = np.array([score for _, score in vector_results])

    # normalize vector scores
    vector_scores = 1 - (vector_scores_raw - vector_scores_raw.min()) / (
        vector_scores_raw.max() - vector_scores_raw.min() + epsilon
    )

    # map doc -> index
    doc_to_idx = {id(doc): i for i, doc in enumerate(all_docs)}
    vector_indices = [doc_to_idx[id(doc)] for doc in vector_docs]

    # --- BM25 (top-k)
    bm25_scores_full = bm25.get_scores(query.split())
    bm25_top_idx = np.argsort(bm25_scores_full)[::-1][:top_k_bm25]

    # --- Union
    candidate_indices = list(set(vector_indices) | set(bm25_top_idx))

    # --- BM25 score
    bm25_scores = np.array([bm25_scores_full[i] for i in candidate_indices])
    bm25_scores = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + epsilon
    )

    # --- Vector score
    vector_score_map = {idx: score for idx, score in zip(vector_indices, vector_scores)}
    vector_scores_final = np.array([
        vector_score_map.get(i, 0.0) for i in candidate_indices
    ])

    # --- Fusion
    combined_scores = alpha * vector_scores_final + (1 - alpha) * bm25_scores

    # --- Rank
    sorted_idx = np.argsort(combined_scores)[::-1]

    top_indices = [candidate_indices[i] for i in sorted_idx[:k]]

    # --- Return docs + embeddings
    top_docs = [all_docs[i] for i in top_indices]
    top_embs = [doc_embeddings[i] for i in top_indices]

    return top_docs, top_embs

def cross_encoder_rerank(
    query: str,
    top_docs: List,
    top_embs: List,   # không dùng trực tiếp nhưng giữ để đồng bộ pipeline
    model,
    k: int = 5,
    batch_size: int = 16
):
    """
    Rerank documents using cross-encoder.

    Args:
        query: str
        top_docs: List[Document]
        top_embs: List[vector] (không dùng nhưng giữ để pipeline thống nhất)
        model: cross-encoder model (đã load sẵn)
        k: số lượng docs trả về
        batch_size: để tránh OOM

    Returns:
        List[Document]
    """

    # 1. Tạo input cho cross-encoder
    pairs = [(query, doc.page_content) for doc in top_docs]

    # 2. Predict theo batch
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_scores = model.predict(batch)
        scores.extend(batch_scores)

    # 3. Sort theo score giảm dần
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    # 4. Lấy top-k
    top_k_docs = [top_docs[i] for i in ranked_indices[:k]]

    return top_k_docs

def adaptive_rerank(
    query: str,
    top_docs: List,
    top_embs: List,
    model,
    k: int,
    rerank_threshold: int = 50,
    rerank_pool_size: int = 100,
    batch_size: int = 16
):
    """
    Adaptive reranking:
    - k nhỏ → rerank toàn bộ
    - k lớn → chỉ rerank top subset
    """

    # Case 1: k nhỏ → full rerank
    if k <= rerank_threshold:
        candidate_docs = top_docs[:rerank_pool_size]

        reranked = cross_encoder_rerank(
            query=query,
            top_docs=candidate_docs,
            top_embs=top_embs[:rerank_pool_size],
            model=model,
            k=k,
            batch_size=batch_size
        )

        return reranked

    # Case 2: k lớn → partial rerank
    else:
        # chỉ rerank top 30 (ví dụ)
        rerank_top_n = min(30, len(top_docs))

        reranked_head = cross_encoder_rerank(
            query=query,
            top_docs=top_docs[:rerank_top_n],
            top_embs=top_embs[:rerank_top_n],
            model=model,
            k=rerank_top_n,
            batch_size=batch_size
        )

        # phần còn lại giữ nguyên
        tail = top_docs[rerank_top_n:k]

        return reranked_head + tail


def company_docs_retrieve(query: str, k: int = 5, alpha: float = 0.5, use_compressor: bool = True):
    """
    Truy vấn thông tin tài liệu của công ty
    """

    # 1. Fusion retrieval → lấy top 50 docs
    top_docs, top_embeddings = fusion_retrieval(
        vectorstore=company_docs_store,
        all_docs=company_docs,
        doc_embeddings=company_embeddings,
        bm25=company_docs_bm25,
        query=query,
        alpha=alpha
    )

    # 2. Embed query
    query_vec = company_embeddings.embed_query(query)

    # 3. Tính similarity query với docs
    sim_query_doc = cosine_similarity([query_vec], top_embeddings)[0]

    # 4. MMR selection
    selected_idx = []
    candidate_idx = list(range(len(top_docs)))

    while len(selected_idx) < k and candidate_idx:
        if len(selected_idx) == 0:
            # chọn doc giống query nhất
            idx = candidate_idx[np.argmax(sim_query_doc[candidate_idx])]
            selected_idx.append(idx)
            candidate_idx.remove(idx)
            continue

        mmr_scores = []
        for i in candidate_idx:
            sim_to_query = sim_query_doc[i]

            sim_to_selected = max(
                cosine_similarity(
                    [top_embeddings[i]],
                    [top_embeddings[j] for j in selected_idx]
                )[0]
            )

            # công thức MMR
            score = alpha * sim_to_query - (1 - alpha) * sim_to_selected
            mmr_scores.append(score)

        best_idx = candidate_idx[np.argmax(mmr_scores)]
        selected_idx.append(best_idx)
        candidate_idx.remove(best_idx)

    selected_docs = [top_docs[i] for i in selected_idx]

    # 🔥 5. Context Compression (NEW)
    if use_compressor:
        cross_encoder = ModelFactory.create(
            model_type="cross_encoder",
            provider="ollama",
            model_name="qwen2.5:7b-instruct-q4"
        )

        selected_docs = cross_encoder.rerank(
            query=query,
            docs=selected_docs,
            top_k=k
        )

    return selected_docs

def retrieve_from_history(history_store, query, embedding_model, top_n=30):
    docs = history_store["docs"]
    embeddings = history_store["embeddings"]

    # embed query
    query_vec = embedding_model.embed_query(query)

    # cosine similarity
    scores = cosine_similarity([query_vec], embeddings)[0]

    # lấy top_n
    top_idx = np.argsort(scores)[-top_n:][::-1]

    top_docs = [docs[i] for i in top_idx]
    top_embs = [embeddings[i] for i in top_idx]

    return top_docs, top_embs


def cv_retrieve(
    subquery: str,
    db_type: str,
    k: int,
    alpha: float,
    history_store  # thêm history vào đây
):
    """
    CV retrieval pipeline:
    - CV_DATABASE → full pipeline (fusion + rerank)
    - HISTORY_CV_DATABASE → lightweight (rerank only)
    """

    reranked_docs = None

    # =========================
    # 1. LOCAL DATABASE
    # =========================
    if db_type == "CV_DATABASE":

        # Step 1: Fusion retrieval
        top_docs, top_embs = fusion_retrieval(
            vectorstore=cv_store,
            all_docs=cv_docs,
            doc_embeddings=cv_embeddings,
            bm25=cv_bm25,
            query=subquery,
            alpha=alpha
        )

        # Step 2: Rerank (Cross Encoder)
        reranked_docs = adaptive_rerank(
            query=subquery,
            top_docs=top_docs,
            top_embs=top_embs,
            model=const.CROSS_ENCODER_LLM,
            k=k
        )


    # =========================
    # 2. HISTORY DATABASE
    # =========================
    elif db_type == "HISTORY_CV_DATABASE":

        if history_store is None:
            return []

        # Step 1: Retrieve trước
        top_docs, top_embs = retrieve_from_history(
            history_store,
            subquery,
            embedding_model = ModelFactory.create(
                model_type="embedding",
                provider="ollama",
                model_name="nomic-embed-text",
            ),
            top_n=30
        )

        # Step 2: Rerank
        reranked_docs = adaptive_rerank(
            query=subquery,
            top_docs=top_docs,
            top_embs=top_embs,
            model= ModelFactory.create(
                model_type="cross_encoder",
                provider="ollama",
                model_name="sam860/qwen3-reranker:0.6b-Q8_0",
            ) ,
            k=k
        )
    # =========================
    # 3. INVALID TYPE
    # =========================
    else:
        raise ValueError(f"Unknown db_type: {db_type}")

    return reranked_docs
    

def general_retrieve(subquery: str, db_type:str, k: int, alpha: float, history_store):
    if db_type == "CV_DATABASE" or db_type == "HISTORY_CV_DATABASE":
        return cv_retrieve(subquery=subquery, db_type=db_type, k = k, history_store = history_store)
    else:
        return company_docs_retrieve(subquery, k, alpha)