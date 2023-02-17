import re
from collections import Counter

def words(text):
    return re.findall(r'\w+', text.lower())

def train(vocab):
    n_words = len(vocab)
    freq = Counter(vocab)
    return freq

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word, vocab):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in vocab)

def known(words, vocab):
    return set(w for w in words if w in vocab)

def correct(word, vocab, n_words):
    candidates = known([word], vocab) or known(edits1(word), vocab) or known_edits2(word, vocab) or [word]
    return max(candidates, key=lambda w: n_words[w])

def spell_check(sentence, vocab):
    n_words = train(vocab)
    corrected_words = []
    for word in sentence.split():
        corrected_words.append(correct(word, vocab, n_words))
    return ' '.join(corrected_words)