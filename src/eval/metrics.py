import re

def normalize(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

def exact_match(gold, pred):
    g, p = normalize(gold), normalize(pred)
    return g in p or p in g

def token_f1(gold, pred):
    g_toks = set(normalize(gold).split())
    p_toks = set(normalize(pred).split())
    if not g_toks or not p_toks:
        return 0.0
    common = g_toks & p_toks
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)
