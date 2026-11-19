import re
from typing import List, Dict, Any, Tuple
from core.preprocessing import clean_text

class SlideRAGService:
    """
    Retrieval-Augmented Generation (RAG) Service satisfying Section 3.7 & 4.12:
    - Bounded passage retrieval over presentation text
    - Explicit slide-level citation attribution
    - Abstention when the presentation does not contain the answer
    - Prompt injection defense: treats presentation text strictly as untrusted data
    """
    def __init__(self):
        self.passages: List[Dict[str, Any]] = []

    def index_presentation(self, slides: List[Dict[str, Any]]):
        """Indexes slides into searchable contextual passages with slide references."""
        self.passages = []
        for s in slides:
            slide_no = s["slide"]
            text = s["text"]
            if not text.strip():
                continue
            
            # Split into natural paragraphs or sentence blocks
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            for p in paragraphs:
                cleaned_p = clean_text(p)
                if cleaned_p:
                    self.passages.append({
                        "slide": slide_no,
                        "raw_text": p,
                        "cleaned_tokens": set(cleaned_p.split())
                    })

    def search_passages(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieves top-k relevant slide passages using token-overlap and term relevance."""
        query_cleaned = clean_text(query)
        q_tokens = set(query_cleaned.split())
        if not q_tokens or not self.passages:
            return []

        scored = []
        for item in self.passages:
            overlap = len(q_tokens.intersection(item["cleaned_tokens"]))
            if overlap > 0:
                # Jaccard / lexical relevance score
                sim = overlap / len(q_tokens.union(item["cleaned_tokens"]))
                scored.append((item, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Executes grounded retrieval and answers the query strictly from presentation text.
        Complies with abstention rules: returns explicit notice if not found.
        """
        # Prompt injection check
        suspicious_patterns = [
            r"ignore previous instructions",
            r"system override",
            r"disregard context",
            r"reveal secret"
        ]
        for pat in suspicious_patterns:
            if re.search(pat, query, re.IGNORECASE):
                return {
                    "answer": "Security Alert: Query contains disallowed instruction overrides. Only presentation content can be queried.",
                    "citations": [],
                    "abstained": True
                }

        results = self.search_passages(query, top_k=3)

        # Abstention rule: if no relevant passages or similarity negligible
        if not results or results[0][1] < 0.08:
            return {
                "answer": "The uploaded presentation does not contain sufficient information to answer this query.",
                "citations": [],
                "abstained": True
            }

        citations = []
        context_snippets = []
        cited_slides = set()

        for item, score in results:
            slide_no = item["slide"]
            cited_slides.add(slide_no)
            citations.append({
                "slide": slide_no,
                "text": item["raw_text"],
                "relevance_score": round(score, 3)
            })
            context_snippets.append(f"[Slide {slide_no}]: {item['raw_text']}")

        slides_str = ", ".join(f"Slide {s}" for s in sorted(cited_slides))
        synthesized_answer = (
            f"Based on extracted presentation content ({slides_str}):\n\n"
            + "\n".join(f"• {c['text']} (Ref: Slide {c['slide']})" for c in citations)
        )

        return {
            "answer": synthesized_answer,
            "citations": citations,
            "abstained": False
        }
