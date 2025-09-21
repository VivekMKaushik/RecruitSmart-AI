import sys
import os
# ensure workspace root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Check for sklearn instead of rank_bm25
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    print('sklearn OK')
except ImportError:
    print('sklearn not available')

# Check for keyword_utils
try:
    import keyword_utils
    print('keyword_utils OK')
except ImportError as e:
    print(f'keyword_utils import error: {e}')

# Check for other essential packages
try:
    import google.generativeai as genai
    print('google-generativeai OK')
except ImportError:
    print('google-generativeai not available')

try:
    import streamlit
    print('streamlit OK')
except ImportError:
    print('streamlit not available')

print('Import check completed!')