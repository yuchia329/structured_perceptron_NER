import sys
import math
from collections import defaultdict
from conlleval import evaluate
import random

##############################################################################
# 1) Data Utilities
##############################################################################
def read_conll_data(filepath):
    """
    Read CoNLL-format data. Return list of sentences, each a list of (word,pos,chunk,ne).
    """
    sentences = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) == 4:
                    word, pos, chunk, ne_tag = parts
                    current.append((word, pos, chunk, ne_tag))
        if current:
            sentences.append(current)
    return sentences

##############################################################################
# 2) Feature Extraction (Full Feature Set)
##############################################################################
def shape(word):
    """
    Returns a shape string (uppercase -> A, lowercase -> a, digits -> d).
    """
    out = []
    for ch in word:
        if ch.isupper():
            out.append('A')
        elif ch.islower():
            out.append('a')
        elif ch.isdigit():
            out.append('d')
        else:
            out.append(ch)
    return "".join(out)

def extract_features(sentence, i, prev_tag, curr_tag):
    """
    Full set of features 1–11 for position i with predicted transition [prev_tag->curr_tag].
    """
    START = "<START>"
    STOP  = "<STOP>"
    feats = {}

    word, pos, chunk, gold_ne = sentence[i]
    wlower = word.lower()
    wshape = shape(word)

    # Helper to get neighbors
    def get_word(k):
        if k < 0:  return START
        if k >= len(sentence): return STOP
        return sentence[k][0]

    def get_pos(k):
        if k < 0:  return START
        if k >= len(sentence): return STOP
        return sentence[k][1]

    # (1) Current word
    feats[f"W_i={word}+T_i={curr_tag}"] = 1.0
    # (2) Previous tag
    feats[f"T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    # (3) Lowercased word
    feats[f"O_i={wlower}+T_i={curr_tag}"] = 1.0
    # (4) POS
    feats[f"P_i={pos}+T_i={curr_tag}"] = 1.0
    # (5) Shape
    feats[f"S_i={wshape}+T_i={curr_tag}"] = 1.0

    # (6) Prev/next word & POS
    pw, pp = get_word(i-1), get_pos(i-1)
    feats[f"W_i-1={pw}+T_i={curr_tag}"] = 1.0
    feats[f"P_i-1={pp}+T_i={curr_tag}"] = 1.0
    if i+1 < len(sentence):
        nw, np = get_word(i+1), get_pos(i+1)
        feats[f"W_i+1={nw}+T_i={curr_tag}"] = 1.0
        feats[f"P_i+1={np}+T_i={curr_tag}"] = 1.0

    # (7) Conjunction (1,3,4) with prev_tag
    feats[f"W_i={word}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    feats[f"O_i={wlower}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    feats[f"P_i={pos}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0

    # (8) Prefixes up to length 4
    for k in range(1,5):
        if len(word) >= k:
            feats[f"PRE_i={word[:k]}+T_i={curr_tag}"] = 1.0

    # (9) Gazetteer membership?  (If you have it, apply here; else skip)
    # feats[f"GAZ_i={something}+T_i={curr_tag}"] = 1.0

    # (10) Capitalization
    starts_cap = word[0].isupper()
    feats[f"CAP_i={starts_cap}+T_i={curr_tag}"] = 1.0

    # (11) Position
    feats[f"POS_i={i+1}+T_i={curr_tag}"] = 1.0

    return feats

##############################################################################
# 3) Viterbi Decoding
##############################################################################
def viterbi_decode(sentence, tag_set, weights):
    """
    Standard Viterbi to find best tag sequence for 'sentence'.
    """
    n = len(sentence)
    if n == 0:
        return []

    viterbi = {}
    backptr = {}

    # i=0
    for tag in tag_set:
        feats = extract_features(sentence, 0, "<START>", tag)
        score = sum(weights.get(f,0.0)*val for f,val in feats.items())
        viterbi[(0,tag)] = score
        backptr[(0,tag)] = None

    # i=1..n-1
    for i in range(1,n):
        for tag in tag_set:
            best_score = float('-inf')
            best_prev  = None
            for ptag in tag_set:
                prev_score = viterbi[(i-1, ptag)]
                feats = extract_features(sentence, i, ptag, tag)
                new_score = prev_score + sum(weights.get(f,0.0)*val for f,val in feats.items())
                if new_score > best_score:
                    best_score = new_score
                    best_prev  = ptag
            viterbi[(i,tag)] = best_score
            backptr[(i,tag)] = best_prev

    # final
    best_score = float('-inf')
    best_tag   = None
    for tag in tag_set:
        sc = viterbi[(n-1,tag)]
        if sc > best_score:
            best_score = sc
            best_tag   = tag

    # backtrack
    out = [best_tag]
    for i in range(n-1,0,-1):
        out.append(backptr[(i,out[-1])])
    out.reverse()
    return out

##############################################################################
# 4) Adagrad Perceptron
##############################################################################
def compute_seq_features(sentence, tag_seq):
    """
    Aggregate features for (sentence, tag_seq).
    """
    feats = defaultdict(float)
    prev_tag = "<START>"
    for i, _ in enumerate(sentence):
        curr_tag = tag_seq[i]
        fdict = extract_features(sentence, i, prev_tag, curr_tag)
        for f, val in fdict.items():
            feats[f] += val
        prev_tag = curr_tag
    return feats

def do_adagrad_update(weights, grad_sumsq, grad, alpha=1.0):
    """
    Adagrad update: 
      grad_sumsq[f] += (grad[f])^2
      weights[f] += alpha * grad[f] / sqrt(grad_sumsq[f] + eps)
    Note the **plus** sign (because grad = gold - pred).
    """
    eps = 1e-8
    for f, g_val in grad.items():
        grad_sumsq[f] += g_val*g_val
        denom = math.sqrt(grad_sumsq[f]) + eps
        # + because in the perceptron, gradient = (gold - pred)
        weights[f] += alpha * g_val / denom

def structured_perceptron_adagrad_train(train_data, dev_data, tag_set,
                                        max_epochs=10, alpha=0.1, early_stop=True):
    """
    Improved Structured Perceptron with Adagrad.
    """
    # Initialize weights randomly
    weights = defaultdict(lambda: random.uniform(-0.01, 0.01))
    grad_sumsq = defaultdict(float)
    best_dev_f = -1.0
    best_weights = None

    for epoch in range(max_epochs):
        print(f"=== Epoch {epoch+1}/{max_epochs} ===")
        random.shuffle(train_data)  # Shuffle for improved convergence

        for sentence in train_data:
            gold_tags = [t[3] for t in sentence]
            pred_tags = viterbi_decode(sentence, tag_set, weights)
            
            if pred_tags != gold_tags:
                gold_vec = compute_seq_features(sentence, gold_tags)
                pred_vec = compute_seq_features(sentence, pred_tags)
                grad = defaultdict(float)
                
                for f, v in gold_vec.items():
                    grad[f] += v
                for f, v in pred_vec.items():
                    grad[f] -= v
                
                # Apply Adagrad update with a smaller alpha for stability
                do_adagrad_update(weights, grad_sumsq, grad, alpha)

        # Evaluate on dev set after each epoch
        dev_p, dev_r, dev_f = evaluate_on_dataset(dev_data, weights, tag_set)
        print(f"Dev Performance -> Precision: {dev_p:.2f}, Recall: {dev_r:.2f}, F1: {dev_f:.2f}")

        # Early stopping condition based on dev performance
        if dev_f > best_dev_f:
            best_dev_f = dev_f
            best_weights = dict(weights)
        elif early_stop:
            print("No improvement detected. Stopping early.")
            break

    return best_weights if best_weights else dict(weights)

def evaluate_on_dataset(dataset, weights, tag_set):
    """
    Decode with standard Viterbi, then run conlleval for chunk-based P/R/F1.
    """
    gold_all = []
    pred_all = []
    for sent in dataset:
        gold = [x[3] for x in sent]
        pred = viterbi_decode(sent, tag_set, weights)
        gold_all.extend(gold)
        pred_all.extend(pred)
    p, r, f1 = evaluate(gold_all, pred_all, verbose=False)
    return p, r, f1

def save_predictions(dataset, weights, tag_set, outfile):
    """
    Write out file with word pos chunk gold pred (5 columns).
    """
    with open(outfile, 'w', encoding='utf-8') as f:
        for sent in dataset:
            gold_tags = [x[3] for x in sent]
            pred_tags = viterbi_decode(sent, tag_set, weights)
            for (w,p,c,g), pr in zip(sent, pred_tags):
                f.write(f"{w} {p} {c} {g} {pr}\n")
            f.write("\n")

##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python adagrad_ner.py <train.conll> <dev.conll> <test.conll>")
        sys.exit(1)

    train_file = sys.argv[1]
    dev_file   = sys.argv[2]
    test_file  = sys.argv[3]

    # 1) Read data
    train_data = read_conll_data(train_file)
    dev_data   = read_conll_data(dev_file)
    test_data  = read_conll_data(test_file)

    # 2) Collect tag set
    tag_set = set()
    for sent in train_data:
        for (_,_,_,ne) in sent:
            tag_set.add(ne)
    tag_set = sorted(tag_set)

    # 3) Train
    alpha = 1.0  # can tune if needed
    print(f"Training structured perceptron with Adagrad, alpha={alpha}")
    best_weights = structured_perceptron_adagrad_train(
        train_data,
        dev_data,
        tag_set,
        max_epochs=10,
        alpha=alpha,
        early_stop=True
    )

    # 4) Evaluate on test
    test_p, test_r, test_f1 = evaluate_on_dataset(test_data, best_weights, tag_set)
    print(f"\nFinal test performance: P={test_p:.2f}, R={test_r:.2f}, F1={test_f1:.2f}")

    # 5) Save predictions
    save_predictions(dev_data, best_weights, tag_set, "preds_dev_adagrad_fixed.txt")
    save_predictions(test_data, best_weights, tag_set, "preds_test_adagrad_fixed.txt")
    print("Wrote preds_dev_adagrad_fixed.txt and preds_test_adagrad_fixed.txt")
