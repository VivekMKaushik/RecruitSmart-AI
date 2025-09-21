def compute_final_score(hard_skill_score, education_score, exp_score, semantic_score, weights=None):
    if weights is None:
        weights = {
            'skill': 0.4,
            'education': 0.2,
            'exp': 0.3, 
            'semantic': 0.1
        }
    final = (weights['skill']*hard_skill_score
             + weights['education']*education_score
             + weights['exp']*exp_score
             + weights['semantic']*semantic_score)
    return round(final, 2)