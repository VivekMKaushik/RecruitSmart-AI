    # embeddings_utils.py

_model = None
_util = None


import os
import logging

def _get_model():
    global _model, _util
    if _model is None:
        # Import here to avoid heavy imports (torch) at module import time
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



def compute_semantic_similarity(jd_text: str, resume_text: str) -> float:
    """
    Return semantic similarity between JD and Resume as a float 0–100.
    Loads model on first use. Supports device/model selection via env.
    Handles errors gracefully and logs them.
    """
    try:
        model, util = _get_model()
        jd_vec = model.encode(jd_text, convert_to_tensor=True)
        resume_vec = model.encode(resume_text, convert_to_tensor=True)
        sim = util.cos_sim(jd_vec, resume_vec).item()
        # scale from -1..1 to 0..100
        return round((sim + 1) / 2 * 100, 2)
    except Exception as e:
        logging.error(f"Semantic similarity error: {e}")
        return 0.0
