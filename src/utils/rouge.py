from collections import Counter

def _tokenize(s):
    return [t for t in s.lower().split() if t]

def _overlap(a, b):
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())

def _f1(p, r, eps=1e-8):
    return (2 * p * r) / max(eps, p + r)


def rouge_n(ref, hyp, n=1):
    r_ngram = [tuple(ref[i:i+n]) for i in range(max(0, len(ref)-n+1))]
    h_ngram = [tuple(hyp[i:i+n]) for i in range(max(0, len(hyp)-n+1))]
    if len(r_ngram) == 0 or len(h_ngram) == 0:
        return 0.0
    inter = _overlap(r_ngram, h_ngram)
    prec = inter / max(1, len(h_ngram))
    rec = inter / max(1, len(r_ngram))
    return _f1(prec, rec)


def rouge_l(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if ref[i] == hyp[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    prec = lcs / max(1, n)
    rec = lcs / max(1, m)
    return _f1(prec, rec)


def ids_to_text(ids, inv_vocab, stop_at="<eos>"):
    toks = []
    for i in ids:
        t = inv_vocab.get(int(i), "")
        if stop_at and t == stop_at:
            break
        if t not in ("<pad>", "<cls>", "<unk>"):
            toks.append(t)
    return " ".join(toks).strip()