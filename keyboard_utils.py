import math
from collections import Counter

def bm25_manual(jd_text: str, resume_text: str, k1=1.5, b=0.75) -> float:
    """
    Compute a BM25-like relevance score of resume against JD without rank_bm25.
    """
    corpus = [jd_text, resume_text]
    N = len(corpus)
    tokenized_corpus = [doc.split() for doc in corpus]
    avgdl = sum(len(doc) for doc in tokenized_corpus) / N
    
    # Document frequencies
    df = {}
    for doc in tokenized_corpus:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1

    # Tokenize resume
    resume_tokens = tokenized_corpus[1]
    score = 0.0
    for word in tokenized_corpus[0]:  # JD words
        f = resume_tokens.count(word)
        if f == 0:
            continue
        idf = math.log((N - df.get(word, 0) + 0.5) / (df.get(word, 0) + 0.5) + 1)
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len(resume_tokens)/avgdl))
    return round(score, 2)
