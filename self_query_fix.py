import re

def extract_legal_references_regex(query):
    """Fast regex fallback - no LLM needed"""
    refs = {"has_references": False}
    
    art = re.search(r'Article\s+(\d+)', query, re.IGNORECASE)
    if art:
        refs["article"] = f"Article {art.group(1)}"
        refs["has_references"] = True
    
    anx = re.search(r'Annex\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII)', query, re.IGNORECASE)
    if anx:
        refs["annex"] = f"Annex {anx.group(1)}"
        refs["has_references"] = True
    
    rec = re.search(r'Recital\s+(\d+)', query, re.IGNORECASE)
    if rec:
        refs["recital"] = f"Recital {rec.group(1)}"
        refs["has_references"] = True
    
    return refs

if __name__ == "__main__":
    tests = [
        "What does Article 9 require for risk management?",
        "Does my system fall under Annex III?",
        "My chatbot answers shipping questions",
        "Article 6 and Annex III classification",
    ]
    for t in tests:
        print(f"  {t}")
        print(f"    -> {extract_legal_references_regex(t)}")
