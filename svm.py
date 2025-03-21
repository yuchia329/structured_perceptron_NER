import sys
import math
import random
from collections import defaultdict
from conlleval import evaluate  # Make sure conlleval.py is present

##############################################################################
# 1) Reading CoNLL data
##############################################################################
def read_conll_data(filepath):
    """
    Reads CoNLL format with 4 columns: word, pos, chunk, ne.
    Returns a list of sentences, each a list of (word, pos, chunk, ne).
    """
    data = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    data.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) == 4:
                    w, p, c, ne = parts
                    current.append((w, p, c, ne))
                else:
                    # Skip malformed lines
                    pass
        if current:
            data.append(current)
    return data

##############################################################################
# 2) Feature Extraction (Full Feature Set)
##############################################################################
def shape(word):
    """Return shape string (uppercase->A, lowercase->a, digit->d)."""
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
    Full set of features (1–11) for token i, transition [prev_tag->curr_tag].
    """
    feats = {}
    START = "<START>"
    STOP  = "<STOP>"

    word, pos, chunk, gold_ne = sentence[i]
    wlower = word.lower()
    wshape = shape(word)

    def get_word(k):
        if k < 0: 
            return START
        elif k >= len(sentence):
            return STOP
        return sentence[k][0]

    def get_pos(k):
        if k < 0:
            return START
        elif k >= len(sentence):
            return STOP
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

    # (6) Prev/Next
    pw, pp = get_word(i-1), get_pos(i-1)
    feats[f"W_i-1={pw}+T_i={curr_tag}"] = 1.0
    feats[f"P_i-1={pp}+T_i={curr_tag}"] = 1.0
    if i+1 < len(sentence):
        nw, np = get_word(i+1), get_pos(i+1)
        feats[f"W_i+1={nw}+T_i={curr_tag}"] = 1.0
        feats[f"P_i+1={np}+T_i={curr_tag}"] = 1.0

    # (7) Conjoin (1,3,4) with prev_tag
    feats[f"W_i={word}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    feats[f"O_i={wlower}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    feats[f"P_i={pos}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0

    # (8) Prefix up to length 4
    for k in range(1,5):
        if len(word) >= k:
            pref = word[:k]
            feats[f"PRE_i={pref}+T_i={curr_tag}"] = 1.0

    # (9) Gazetteer membership? (If you have it, add here)
    # e.g. feats[f"GAZ={in_gaz}+T_i={curr_tag}"] = 1.0

    # (10) Capitalization
    is_cap = word[0].isupper()
    feats[f"CAP_i={is_cap}+T_i={curr_tag}"] = 1.0

    # (11) Position
    feats[f"POS_i={i+1}+T_i={curr_tag}"] = 1.0

    return feats

##############################################################################
# 3) Standard Viterbi for final predictions
##############################################################################
def viterbi_decode(sentence, tag_set, weights):
    """
    Standard Viterbi decoding for evaluation (no cost).
    """
    n = len(sentence)
    if n == 0:
        return []
    viterbi = {}
    backptr = {}

    # init
    for tag in tag_set:
        fdict = extract_features(sentence, 0, "<START>", tag)
        score = sum(weights.get(f,0.0)*val for f,val in fdict.items())
        viterbi[(0,tag)] = score
        backptr[(0,tag)] = None

    # fill
    for i in range(1,n):
        for tag in tag_set:
            best_score = float('-inf')
            best_prev  = None
            for ptag in tag_set:
                prev_score = viterbi[(i-1, ptag)]
                fdict = extract_features(sentence, i, ptag, tag)
                step_score = prev_score + sum(weights.get(f,0.0)*val for f,val in fdict.items())
                if step_score > best_score:
                    best_score = step_score
                    best_prev  = ptag
            viterbi[(i,tag)] = best_score
            backptr[(i,tag)] = best_prev

    # final
    best_tag = None
    best_score = float('-inf')
    for tag in tag_set:
        sc = viterbi[(n-1,tag)]
        if sc > best_score:
            best_score = sc
            best_tag   = tag

    # backtrack
    path = [best_tag]
    for i in range(n-1,0,-1):
        path.append(backptr[(i,path[-1])])
    path.reverse()
    return path

##############################################################################
# 4) Cost-Augmented Viterbi for Structured SVM
##############################################################################
def cost_augmented_viterbi(sentence, tag_set, weights, gold_tags, cost_multiplier=10):
    """
    Cost-augmented decoding:
      score(y') = w . Phi(x, y') + cost(y')
    where cost is (# mismatches * cost_multiplier).
    """
    n = len(sentence)
    if n == 0:
        return []

    viterbi = {}
    backptr = {}

    # i=0
    for tag in tag_set:
        feats = extract_features(sentence, 0, "<START>", tag)
        base_score = sum(weights.get(f,0.0)*val for f,val in feats.items())
        mismatch_cost = cost_multiplier if (tag != gold_tags[0]) else 0
        viterbi[(0,tag)] = base_score + mismatch_cost
        backptr[(0,tag)] = None

    # i=1..n-1
    for i in range(1,n):
        for tag in tag_set:
            best_score = float('-inf')
            best_prev  = None
            for ptag in tag_set:
                prev_score = viterbi[(i-1,ptag)]
                feats = extract_features(sentence, i, ptag, tag)
                step_score = prev_score + sum(weights.get(f,0.0)*val for f,val in feats.items())
                if tag != gold_tags[i]:
                    step_score += cost_multiplier
                if step_score > best_score:
                    best_score = step_score
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
    path = [best_tag]
    for i in range(n-1,0,-1):
        path.append(backptr[(i,path[-1])])
    path.reverse()
    return path

##############################################################################
# 5) Helpers: Aggregating Features, Evaluating
##############################################################################
def get_sequence_features(sentence, tag_seq):
    """
    Compute aggregated features for (sentence, tag_seq).
    """
    out = defaultdict(float)
    prev_tag = "<START>"
    for i, _ in enumerate(sentence):
        curr_tag = tag_seq[i]
        fdict = extract_features(sentence, i, prev_tag, curr_tag)
        for f,val in fdict.items():
            out[f] += val
        prev_tag = curr_tag
    return out

def evaluate_on_dataset(dataset, weights, tag_set):
    """
    Use standard Viterbi for each sentence, then compute P/R/F1 chunk metrics.
    """
    gold_all = []
    pred_all = []
    for sent in dataset:
        gold_tags = [tok[3] for tok in sent]
        pred_tags = viterbi_decode(sent, tag_set, weights)
        gold_all.extend(gold_tags)
        pred_all.extend(pred_tags)
    p, r, f1 = evaluate(gold_all, pred_all, verbose=False)
    return p, r, f1

def save_predictions(dataset, weights, tag_set, outfile):
    """
    Write 5-column output: word pos chunk gold pred
    """
    with open(outfile,'w',encoding='utf-8') as f:
        for sent in dataset:
            gold_tags = [x[3] for x in sent]
            pred_tags = viterbi_decode(sent, tag_set, weights)
            for (w,p,c,g), pr in zip(sent, pred_tags):
                f.write(f"{w} {p} {c} {g} {pr}\n")
            f.write("\n")

##############################################################################
# 6) Structured SVM with Adagrad + L2
##############################################################################
def structured_svm_adagrad_train(
    train_data,
    dev_data,
    tag_set,
    cost_multiplier=10,
    reg_lambda=1e-4,
    step_size=0.1,
    max_epochs=10,
    early_stop=True
):
    """
    Structured SVM training:
      - cost-augmented decoding with Hamming distance * cost_multiplier
      - subgradient approach: margin check, then update if violation
      - L2 regularization
      - Adagrad for each feature dimension
      - early stopping on dev F1

    grad_sumsq[f] accumulates squared gradient for feature f.
    The update is:
      1) L2 shrink: w[f] *= (1 - step_size * reg_lambda)
      2) if margin violated, do Adagrad step with grad = (gold - pred).
    """
    weights = defaultdict(float)
    grad_sumsq = defaultdict(float)
    best_weights = None
    best_dev_f = -1.0

    def l2_shrink(w):
        decay = 1.0 - step_size*reg_lambda
        for f in list(w.keys()):
            w[f] = decay * w[f]

    def adagrad_update(grad):
        eps = 1e-8
        for f, gval in grad.items():
            grad_sumsq[f] += gval*gval
            w[f] += (step_size * gval) / (math.sqrt(grad_sumsq[f]) + eps)

    w = weights  # alias for brevity

    for epoch in range(max_epochs):
        print(f"=== Epoch {epoch+1}/{max_epochs} (S-SVM + Adagrad) ===")
        # random.shuffle(train_data)  # optional

        for sentence in train_data:
            gold_tags = [x[3] for x in sentence]
            # 1) cost-aug decode
            pred_tags = cost_augmented_viterbi(sentence, tag_set, w, gold_tags, cost_multiplier)
            # 2) margin check
            # cost(pred) = cost_multiplier * (# mismatches)
            cost_pred = sum(1 for a,b in zip(gold_tags, pred_tags) if a!=b) * cost_multiplier
            # w.dot(gold) vs w.dot(pred)
            gold_phi = get_sequence_features(sentence, gold_tags)
            pred_phi = get_sequence_features(sentence, pred_tags)
            w_gold = sum(w.get(f,0.0)*val for f,val in gold_phi.items())
            w_pred = sum(w.get(f,0.0)*val for f,val in pred_phi.items())
            lhs = w_gold + 0.0  # cost(gold)=0
            rhs = w_pred + cost_pred

            # 3) If margin violated => update
            #    Always do L2 shrink first
            l2_shrink(w)

            if lhs < rhs - 1e-12:
                # subgradient = (gold_phi - pred_phi)
                grad = defaultdict(float)
                for f,v in gold_phi.items():
                    grad[f] += v
                for f,v in pred_phi.items():
                    grad[f] -= v
                # adagrad update
                adagrad_update(grad)
            else:
                # no violation => no gradient, but we already did L2 shrink
                pass

        # Evaluate on dev
        dev_p, dev_r, dev_f = evaluate_on_dataset(dev_data, w, tag_set)
        print(f"  Dev -> P={dev_p:.2f}, R={dev_r:.2f}, F1={dev_f:.2f}")
        if dev_f > best_dev_f:
            best_dev_f = dev_f
            best_weights = dict(w)
        elif early_stop:
            print("  No improvement; early stopping.")
            break

    return best_weights if best_weights is not None else dict(w)

##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python structured_svm_adagrad.py <train.conll> <dev.conll> <test.conll>")
        sys.exit(1)

    train_file = sys.argv[1]
    dev_file   = sys.argv[2]
    test_file  = sys.argv[3]

    # 1) Read data
    train_data = read_conll_data(train_file)
    dev_data   = read_conll_data(dev_file)
    test_data  = read_conll_data(test_file)

    # 2) Collect tag set
    all_tags = set()
    for sent in train_data:
        for (_,_,_,ne) in sent:
            all_tags.add(ne)
    tag_set = sorted(all_tags)

    # 3) Hyperparameters (tune these!)
    step_size     = 0.1
    reg_lambda    = 1e-4
    cost_mult     = 10
    max_epochs    = 10
    early_stop    = True

    print(f"Training Structured SVM + Adagrad: step_size={step_size}, reg_lambda={reg_lambda}, cost={cost_mult}")
    best_weights = structured_svm_adagrad_train(
        train_data,
        dev_data,
        tag_set,
        cost_multiplier=cost_mult,
        reg_lambda=reg_lambda,
        step_size=step_size,
        max_epochs=max_epochs,
        early_stop=early_stop
    )

    # Evaluate final on dev & test
    dev_p, dev_r, dev_f1   = evaluate_on_dataset(dev_data,  best_weights, tag_set)
    test_p, test_r, test_f1 = evaluate_on_dataset(test_data, best_weights, tag_set)
    print("\nFinal performance with best dev weights:")
    print(f" Dev  -> P={dev_p:.2f}, R={dev_r:.2f}, F1={dev_f1:.2f}")
    print(f" Test -> P={test_p:.2f}, R={test_r:.2f}, F1={test_f1:.2f}")

    # Save predictions
    save_predictions(dev_data,  best_weights, tag_set, "preds_dev_svm_adagrad.txt")
    save_predictions(test_data, best_weights, tag_set, "preds_test_svm_adagrad.txt")
    print("Wrote preds_dev_svm_adagrad.txt and preds_test_svm_adagrad.txt")
