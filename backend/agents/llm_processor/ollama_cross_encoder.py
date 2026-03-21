from typing import List, Tuple
from langchain_core.documents import Document

class BaseCrossEncoder:
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError

    def rerank(self, query: str, docs: List[Document], top_k: int = 5):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.predict(pairs)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [docs[i] for i in ranked_indices[:top_k]]

    
class OllamaCrossEncoder(BaseCrossEncoder):
    def __init__(self, llm):
        self.llm = llm

    def predict(self, pairs):
        scores = []

        for query, doc in pairs:
            prompt = f"""<|query|>
            {query}
            <|document|>
            {doc}
            <|score|>"""

            res = self.llm.invoke(prompt).content.strip()

            try:
                score = float(res)
            except:
                score = 0.0

            scores.append(score)

        return scores