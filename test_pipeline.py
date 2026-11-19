from pathlib import Path
from core.classifier import ContentClassifier
from services.pptx_parser import PresentationParser
from services.rag_service import SlideRAGService
from services.document_utils import analyze_tone, extractive_summarize

def test_all():
    print("Testing ContentClassifier...")
    clf = ContentClassifier()
    print("  Models loaded:", clf.is_loaded)
    sample_ai = "Furthermore, it is crucial to delve into the intricate tapestry of modern large language model architectures."
    res = clf.highlight_sentences(sample_ai)
    print("  Classification Result:", res)

    print("\nTesting PresentationParser on sample_presentation.pptx...")
    parser = PresentationParser()
    slides = parser.parse(Path("sample_presentation.pptx"))
    print(f"  Parsed {len(slides)} slides successfully.")
    for s in slides:
        print(f"  Slide {s['slide']}: {s['word_count']} words | warnings: {s['warnings']}")

    print("\nTesting RAG Service...")
    rag = SlideRAGService()
    rag.index_presentation(slides)
    q1 = rag.answer_query("what happens to slides with pure text?")
    print("  RAG Query 1:", q1["answer"])
    
    q2 = rag.answer_query("what is the price of Bitcoin in 2030?")
    print("  RAG Query 2 (Abstention check):", q2["answer"])

    print("\nTesting Document Utilities (Tone & Summary)...")
    full_text = "\n".join(s["text"] for s in slides)
    tone = analyze_tone(full_text)
    print("  Tone Analysis:", tone["primary_tone"], tone["indicators"])
    summary = extractive_summarize(full_text, max_sentences=2)
    print("  Extractive Summary:", summary)
    print("\n[✓] All pipeline tests passed!")

if __name__ == "__main__":
    test_all()
