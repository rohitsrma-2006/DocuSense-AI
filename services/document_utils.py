import re
from typing import List, Dict, Any
from core.preprocessing import tokenize_sentences, clean_text

# Transparent lexicon sets for Tone Analysis (Section 3.7)
TONE_LEXICONS = {
    "academic": {
        "furthermore", "moreover", "therefore", "methodology", "hypothesis", 
        "empirical", "corpus", "classification", "theoretical", "statistical",
        "evaluation", "investigate", "framework", "synthesize", "perspective"
    },
    "analytical": {
        "data", "metrics", "accuracy", "precision", "recall", "performance",
        "calculate", "threshold", "latency", "benchmark", "probability", "vector",
        "distribution", "variance", "correlation", "gradient", "loss", "frequency"
    },
    "promotional": {
        "revolutionary", "groundbreaking", "ultimate", "perfect", "seamless",
        "unmatched", "world-class", "transformative", "incredible", "effortless",
        "extraordinary", "cutting-edge", "game-changing", "astonishing"
    },
    "hedged_cautious": {
        "suggests", "indicates", "potentially", "likely", "estimated", "approximate",
        "preliminary", "appears", "conditional", "requires", "caution", "review"
    }
}

def analyze_tone(text: str) -> Dict[str, Any]:
    """
    Transparent lexicon-based tone analyzer complying with Section 3.7:
    Detects proportions of academic, analytical, promotional, and cautious vocabulary.
    """
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    if not words:
        return {"primary_tone": "Neutral", "breakdown": {}, "indicators": []}

    total_words = len(words)
    scores = {}
    found_indicators = []

    for category, lexicon in TONE_LEXICONS.items():
        matched = [w for w in words if w in lexicon]
        count = len(matched)
        pct = round((count / total_words) * 100, 1)
        scores[category] = pct
        if matched:
            found_indicators.append(f"{category.capitalize()}: {', '.join(set(matched[:4]))}")

    # Determine primary tone
    sorted_tones = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_tones[0][1] > 0.5:
        primary = sorted_tones[0][0].replace("_", " ").capitalize()
    else:
        primary = "Standard Informational"

    return {
        "primary_tone": primary,
        "scores": scores,
        "indicators": found_indicators
    }

def extractive_summarize(text: str, max_sentences: int = 3) -> List[str]:
    """
    Extractive summarization using frequency-weighted sentence ranking (Section 3.7).
    Selects top salient sentences while preserving narrative document order.
    """
    sentences = tokenize_sentences(text or "")
    if len(sentences) <= max_sentences:
        return [s.strip() for s in sentences if s.strip()]

    # Calculate word frequency over cleaned vocabulary
    word_freq = {}
    for sent in sentences:
        tokens = clean_text(sent).split()
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1

    if not word_freq:
        return sentences[:max_sentences]

    max_f = max(word_freq.values())
    for w in word_freq:
        word_freq[w] = word_freq[w] / max_f

    # Score sentences based on normalized token weights
    ranked = []
    for i, sent in enumerate(sentences):
        tokens = clean_text(sent).split()
        if not tokens:
            continue
        score = sum(word_freq.get(t, 0) for t in tokens) / max(1, len(tokens))
        ranked.append((i, sent.strip(), score))

    # Pick top scoring sentences
    top_ranked = sorted(ranked, key=lambda x: x[2], reverse=True)[:max_sentences]
    # Re-sort into original document appearance order
    top_ranked.sort(key=lambda x: x[0])

    return [item[1] for item in top_ranked]
