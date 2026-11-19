import re
import unicodedata

# Default standard English stopwords fallback in case NLTK corpus needs download
DEFAULT_ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now"
}

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = DEFAULT_ENGLISH_STOPWORDS

def tokenize_words(text: str):
    """
    Splits text into word punctuation tokens.
    Uses nltk.tokenize.wordpunct_tokenize if available, else regex tokenization.
    """
    try:
        from nltk.tokenize import wordpunct_tokenize
        return wordpunct_tokenize(text)
    except Exception:
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def tokenize_sentences(text: str):
    """
    Splits document into sentences.
    Uses nltk.tokenize.sent_tokenize if available, else clean regex sentence boundaries.
    """
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        try:
            return sent_tokenize(text or "")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return sent_tokenize(text or "")
    except Exception:
        # Robust regex sentence boundary splitter fallback
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text or "")
        return [s.strip() for s in sentences if s.strip()]

def clean_text(value: str) -> str:
    """
    Strict NLP preprocessing pipeline specified in PDF Section 4.2:
    1. Unicode normalisation (NFKC)
    2. Lower-casing
    3. Removal of URL patterns (https?://\S+|www\.\S+)
    4. Conservative character filtering ([^a-z0-9\s'])
    5. Tokenization and selective stop-word handling
    6. Preservation of tokens with length > 1
    """
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = tokenize_words(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)
