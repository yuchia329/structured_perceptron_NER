import sys
import math
import random
from collections import defaultdict
from conlleval import evaluate

##################################################
# 1) Data Utilities
##################################################
def read_conll_data(filepath):
    """
    Reads CoNLL-format data from 'filepath' and returns a list of sentences.
    Each sentence is a list of (word, pos, chunk, ne_tag).
    """
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) == 4:
                    word, pos, chunk, ne = parts
                    current_sentence.append((word, pos, chunk, ne))
                else:
                    # Skip any malformed lines
                    pass
        if current_sentence:
            sentences.append(current_sentence)

    return sentences


##################################################
# 2) Feature Extraction
##################################################
def shape(word):
    """
    Returns the 'shape' of a word:
      - uppercase -> 'A'
      - lowercase -> 'a'
      - digits    -> 'd'
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

def is_in_gazetteer(word, tag, gazetteers):
    if gazetteers is None:
        return False
    if tag not in gazetteers:
        return False
    return word in gazetteers[tag]

def extract_features(sentence, i, prev_tag, curr_tag,
                     gazetteers=None, limited_features=False):
    """
    Extracts features for the i-th token in 'sentence', given
    the previous tag 'prev_tag' and current tag 'curr_tag'.

    If limited_features=True, ONLY features 1-4 are used:
      1. W_i
      2. prev_tag
      3. lowercased word
      4. POS
    Otherwise, uses the full set (1-11).
    """
    feats = {}
    word, pos, chunk, gold_ne = sentence[i]
    START = "<START>"
    STOP = "<STOP>"

    # Helper functions for neighbor tokens
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

    ######################
    # Features 1–4
    ######################
    # (1) Current word
    feats[f"W_i={word}+T_i={curr_tag}"] = 1.0
    # (2) Previous tag
    feats[f"T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    # (3) Lowercased word
    lower_w = word.lower()
    feats[f"O_i={lower_w}+T_i={curr_tag}"] = 1.0
    # (4) POS
    feats[f"P_i={pos}+T_i={curr_tag}"] = 1.0

    if limited_features:
        # If we are using ONLY features 1–4, return now.
        return feats

    ######################
    # Extended features (5–11)
    ######################
    # (5) Word shape
    w_shape = shape(word)
    feats[f"S_i={w_shape}+T_i={curr_tag}"] = 1.0

    # (6) Features (1–4) for previous and next word
    prev_word = get_word(i-1)
    prev_pos = get_pos(i-1)
    feats[f"W_i-1={prev_word}+T_i={curr_tag}"] = 1.0
    feats[f"P_i-1={prev_pos}+T_i={curr_tag}"] = 1.0

    # next token
    next_word = get_word(i+1)
    next_pos = get_pos(i+1)
    if i + 1 < len(sentence):
        feats[f"W_i+1={next_word}+T_i={curr_tag}"] = 1.0
        feats[f"P_i+1={next_pos}+T_i={curr_tag}"] = 1.0

    # (7) Conjoin (1, 3, 4) with the previous tag
    feats[f"W_i={word}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    feats[f"O_i={lower_w}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    feats[f"P_i={pos}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0

    # (8) Prefix for lengths [1..4]
    for k in range(1, 5):
        if len(word) >= k:
            prefix = word[:k]
            feats[f"PRE_i={prefix}+T_i={curr_tag}"] = 1.0

    # (9) Gazetteer membership
    gaz_in = is_in_gazetteer(word, curr_tag, gazetteers)
    feats[f"GAZ_i={gaz_in}+T_i={curr_tag}"] = 1.0

    # (10) Capitalization
    is_cap = word[0].isupper()
    feats[f"CAP_i={is_cap}+T_i={curr_tag}"] = 1.0

    # (11) Position (1-based)
    position = i + 1
    feats[f"POS_i={position}+T_i={curr_tag}"] = 1.0

    return feats


##################################################
# 3) Viterbi Decoder
##################################################
def viterbi_decode(sentence, tag_set, weights, gazetteers=None, limited_features=False):
    """
    Standard Viterbi. Returns best tag sequence for the sentence.
    """
    n = len(sentence)
    if n == 0:
        return []
    viterbi = {}
    backptr = {}

    # Initialize
    for tag in tag_set:
        feats = extract_features(sentence, 0, "<START>", tag,
                                 gazetteers, limited_features=limited_features)
        score = sum(weights.get(f, 0.0)*val for f,val in feats.items())
        viterbi[(0,tag)] = score
        backptr[(0,tag)] = None

    # Fill
    for i in range(1, n):
        for tag in tag_set:
            best_score = float('-inf')
            best_prev = None
            for prev_tag in tag_set:
                prev_score = viterbi[(i-1, prev_tag)]
                feats = extract_features(sentence, i, prev_tag, tag,
                                         gazetteers, limited_features=limited_features)
                score = prev_score + sum(weights.get(f, 0.0)*val for f,val in feats.items())
                if score > best_score:
                    best_score = score
                    best_prev = prev_tag
            viterbi[(i, tag)] = best_score
            backptr[(i, tag)] = best_prev

    # Final
    best_final_score = float('-inf')
    best_final_tag = None
    for tag in tag_set:
        score = viterbi[(n-1, tag)]
        if score > best_final_score:
            best_final_score = score
            best_final_tag = tag

    # Reconstruct
    best_path = [best_final_tag]
    for i in range(n-1, 0, -1):
        best_path.append(backptr[(i, best_path[-1])])
    best_path.reverse()
    return best_path


##################################################
# 4) Structured Perceptron Training + Evaluate
##################################################
def compute_sequence_features(sentence, tag_seq, gazetteers=None, weights=None, limited_features=False):
    """
    Compute the aggregated feature vector for an entire sentence + predicted tag sequence.
    Optionally compute total score under 'weights'.
    """
    feat_dict = defaultdict(float)
    total_score = 0.0
    prev_tag = "<START>"

    for i in range(len(sentence)):
        curr_tag = tag_seq[i]
        feats = extract_features(sentence, i, prev_tag, curr_tag,
                                 gazetteers, limited_features=limited_features)
        for f, val in feats.items():
            feat_dict[f] += val
            if weights:
                total_score += weights.get(f,0.0)*val
        prev_tag = curr_tag

    return feat_dict, total_score

def structured_perceptron_train(train_data, dev_data, tag_set,
                                gazetteers=None, max_epochs=5, early_stop=True,
                                limited_features=False):
    """
    Trains a structured perceptron model with the chosen feature set (limited or full).
    We do early stopping on dev set using CoNLL chunk F1.
    """
    weights = defaultdict(float)
    best_weights = None
    best_dev_f1 = -1.0

    def update(weights, gold_feats, pred_feats):
        for f,v in gold_feats.items():
            weights[f] += v
        for f,v in pred_feats.items():
            weights[f] -= v

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs} (limited={limited_features}):")
        # random.shuffle(train_data)  # optional
        for sentence in train_data:
            gold_tags = [x[3] for x in sentence]
            pred_tags = viterbi_decode(sentence, tag_set, weights,
                                       gazetteers=gazetteers,
                                       limited_features=limited_features)
            if pred_tags != gold_tags:
                # compute feature difference
                gold_f, _ = compute_sequence_features(sentence, gold_tags,
                                                      gazetteers, None, limited_features)
                pred_f, _ = compute_sequence_features(sentence, pred_tags,
                                                      gazetteers, None, limited_features)
                update(weights, gold_f, pred_f)

        # Evaluate on dev
        p, r, f1 = evaluate_on_dataset(dev_data, weights, tag_set,
                                       gazetteers=gazetteers,
                                       limited_features=limited_features)
        print(f"  Dev -> P={p:.2f}, R={r:.2f}, F1={f1:.2f}")

        # Early stopping
        if f1 > best_dev_f1:
            best_dev_f1 = f1
            best_weights = dict(weights)
        elif early_stop:
            print("  No improvement; early stopping.")
            break

    if best_weights is None:
        return dict(weights)
    return best_weights

def evaluate_on_dataset(dataset, weights, tag_set, gazetteers=None, limited_features=False):
    """
    Use conlleval-based chunk evaluation on 'dataset'.
    """
    gold_all = []
    pred_all = []
    for sentence in dataset:
        gold_tags = [x[3] for x in sentence]
        pred_tags = viterbi_decode(sentence, tag_set, weights,
                                   gazetteers=gazetteers,
                                   limited_features=limited_features)
        gold_all.extend(gold_tags)
        pred_all.extend(pred_tags)
    p, r, f1 = evaluate(gold_all, pred_all, verbose=False)
    return p, r, f1


##################################################
# 5) Putting It All Together / Main
##################################################
def save_predictions(dataset, weights, tag_set, out_path,
                     gazetteers=None, limited_features=False):
    """
    Write a file with 5 columns: word, pos, chunk, gold, pred
    """
    with open(out_path, 'w', encoding='utf-8') as outf:
        for sentence in dataset:
            gold_tags = [x[3] for x in sentence]
            pred_tags = viterbi_decode(sentence, tag_set, weights,
                                       gazetteers=gazetteers,
                                       limited_features=limited_features)
            for (w,p,c,g), pred in zip(sentence, pred_tags):
                outf.write(f"{w} {p} {c} {g} {pred}\n")
            outf.write("\n")


def print_feature_extremes(weights, top_n=5):
    """
    Print the top-n highest-weight features and top-n lowest-weight features.
    """
    # Sort (feature, weight) by weight
    items = list(weights.items())
    items.sort(key=lambda x: x[1], reverse=True)
    print("Top {} features:".format(top_n))
    for f, w in items[:top_n]:
        print(f"  {f}: {w:.4f}")
    print("Bottom {} features:".format(top_n))
    for f, w in items[-top_n:]:
        print(f"  {f}: {w:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ner_experiments.py <train_file> <dev_file> <test_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    dev_file   = sys.argv[2]
    test_file  = sys.argv[3]

    # 1) Read data
    train_data = read_conll_data(train_file)
    dev_data   = read_conll_data(dev_file)
    test_data  = read_conll_data(test_file)

    # 2) Collect NE tag set
    all_tags = set()
    for sent in train_data:
        for (w,p,c,ne) in sent:
            all_tags.add(ne)
    tag_set = sorted(all_tags)

    # Optional gazetteer
    gazetteers = None

    print("==================================================")
    print("Training with LIMITED FEATURE SET (features 1-4)")
    weights_limited = structured_perceptron_train(
        train_data, dev_data, tag_set,
        gazetteers=gazetteers, max_epochs=10, early_stop=True,
        limited_features=True
    )
    # Evaluate on dev/test
    dev_p, dev_r, dev_f1 = evaluate_on_dataset(dev_data, weights_limited, tag_set,
                                               gazetteers, limited_features=True)
    test_p, test_r, test_f1 = evaluate_on_dataset(test_data, weights_limited, tag_set,
                                                  gazetteers, limited_features=True)
    print(f"[LIMITED] Dev  -> P={dev_p:.2f}, R={dev_r:.2f}, F1={dev_f1:.2f}")
    print(f"[LIMITED] Test -> P={test_p:.2f}, R={test_r:.2f}, F1={test_f1:.2f}")

    # Save dev predictions for manual error analysis
    save_predictions(dev_data, weights_limited, tag_set, "preds_dev_limited.txt",
                     gazetteers=gazetteers, limited_features=True)
    print("Wrote dev predictions (limited features) to preds_dev_limited.txt")

    # Save test predictions for reference
    save_predictions(test_data, weights_limited, tag_set, "preds_test_limited.txt",
                     gazetteers=gazetteers, limited_features=True)
    print("Wrote test predictions (limited features) to preds_test_limited.txt")

    print("\n==================================================")
    print("Training with FULL FEATURE SET (features 1-11)")
    weights_full = structured_perceptron_train(
        train_data, dev_data, tag_set,
        gazetteers=gazetteers, max_epochs=10, early_stop=True,
        limited_features=False
    )
    # Evaluate on dev/test
    dev_p_f, dev_r_f, dev_f1_f = evaluate_on_dataset(dev_data, weights_full, tag_set,
                                                     gazetteers, limited_features=False)
    test_p_f, test_r_f, test_f1_f = evaluate_on_dataset(test_data, weights_full, tag_set,
                                                        gazetteers, limited_features=False)
    print(f"[FULL] Dev  -> P={dev_p_f:.2f}, R={dev_r_f:.2f}, F1={dev_f1_f:.2f}")
    print(f"[FULL] Test -> P={test_p_f:.2f}, R={test_r_f:.2f}, F1={test_f1_f:.2f}")

    # Save dev predictions for manual error analysis
    save_predictions(dev_data, weights_full, tag_set, "preds_dev_full.txt",
                     gazetteers=gazetteers, limited_features=False)
    print("Wrote dev predictions (full features) to preds_dev_full.txt")

    # Save test predictions for reference
    save_predictions(test_data, weights_full, tag_set, "preds_test_full.txt",
                     gazetteers=gazetteers, limited_features=False)
    print("Wrote test predictions (full features) to preds_test_full.txt")

    # Print the top 5 and bottom 5 features for the full model
    print("\n==================================================")
    print("Inspecting top/bottom features in FULL model:")
    print_feature_extremes(weights_full, top_n=5)

    # Optionally, save the final full model weights to a file
    with open("final_full_model_weights_ner.txt", "w", encoding="utf-8") as f:
        for feat, val in sorted(weights_full.items(), key=lambda x:x[1], reverse=True):
            f.write(f"{feat}\t{val}\n")
    print("Saved full model weights to final_full_model_weights_ner.txt")
