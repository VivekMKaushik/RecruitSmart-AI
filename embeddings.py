# embeddings.py

import os
import logging

_model = None
_util = None

def _get_model():
    global _model, _util
    if _model is None:
        from sentence_transformers import SentenceTransformer, util
        model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
        device = os.getenv("EMBEDDINGS_DEVICE", "cpu")
        try:
            _model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        _util = util
    return _model, _util

def semantic_score(jd_text, resume_text):
    """
    Compute semantic similarity score (0-100) between JD and Resume.
    Uses environment for model/device selection. Handles errors gracefully.
    """
    try:
        model, util = _get_model()
        jd_vec = model.encode(jd_text, convert_to_tensor=True)
        resume_vec = model.encode(resume_text, convert_to_tensor=True)
        sim = util.cos_sim(jd_vec, resume_vec).item()
        return round((sim + 1) / 2 * 100, 2)
    except Exception as e:
        logging.error(f"Semantic score error: {e}")
        return 0.0
