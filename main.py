import sys
import math
import random
from collections import defaultdict
from conlleval import evaluate  # <-- import the official CoNLL evaluation

# -------------------------------------------------------------------------
# 1. Data utilities
# -------------------------------------------------------------------------
def read_conll_data(filepath):
    """
    Reads CoNLL-format data from 'filepath' and returns a list of sentences.
    Each sentence is represented as a list of tuples: (word, pos, chunk, ne_tag).
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
                # Expecting 4 columns: token, pos, chunk, ne_tag
                if len(parts) == 4:
                    word, pos, chunk, ne_tag = parts
                    current_sentence.append((word, pos, chunk, ne_tag))
                else:
                    # If a line is malformed, skip it or handle appropriately
                    pass

        if current_sentence:
            sentences.append(current_sentence)

    return sentences

# -------------------------------------------------------------------------
# 2. Feature Extraction
# -------------------------------------------------------------------------

def shape(word):
    """
    Returns the 'shape' of a word:
     - all uppercase letters => 'A'
     - all lowercase letters => 'a'
     - all digits => 'd'
    """
    s = []
    for ch in word:
        if ch.isupper():
            s.append('A')
        elif ch.islower():
            s.append('a')
        elif ch.isdigit():
            s.append('d')
        else:
            s.append(ch)
    return "".join(s)

def is_in_gazetteer(word, tag, gazetteers):
    """
    Check if 'word' is in the gazetteer for 'tag'.
    gazetteers: dict of tag -> set_of_words.
    """
    if gazetteers is None:
        return False
    if tag not in gazetteers:
        return False
    return word in gazetteers[tag]

def extract_features(sentence, i, prev_tag, curr_tag, gazetteers=None):
    """
    Extract features for position i in 'sentence', given the previous tag (prev_tag)
    and the current tag (curr_tag). Return a dict {feature_name: value, ...}.
    """
    START = "<START>"
    STOP = "<STOP>"

    word, pos, chunk, gold_tag = sentence[i]
    lower_word = word.lower()
    word_shape = shape(word)

    feats = {}

    # (1) Current word
    feats[f"W_i={word}+T_i={curr_tag}"] = 1.0

    # (2) Previous tag
    feats[f"T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0

    # (3) Lowercased word
    feats[f"O_i={lower_word}+T_i={curr_tag}"] = 1.0

    # (4) Current POS tag
    feats[f"P_i={pos}+T_i={curr_tag}"] = 1.0

    # (5) Shape of current word
    feats[f"S_i={word_shape}+T_i={curr_tag}"] = 1.0

    def get_word(idx):
        if idx < 0:
            return START
        elif idx >= len(sentence):
            return STOP
        return sentence[idx][0]

    def get_pos(idx):
        if idx < 0:
            return START
        elif idx >= len(sentence):
            return STOP
        return sentence[idx][1]

    # (6) Features 1-4 for previous and next word
    prev_w = get_word(i - 1)
    prev_pos = get_pos(i - 1)
    feats[f"W_i-1={prev_w}+T_i={curr_tag}"] = 1.0
    feats[f"P_i-1={prev_pos}+T_i={curr_tag}"] = 1.0

    next_w = get_word(i + 1)
    next_pos = get_pos(i + 1)
    if i + 1 < len(sentence):
        feats[f"W_i+1={next_w}+T_i={curr_tag}"] = 1.0
        feats[f"P_i+1={next_pos}+T_i={curr_tag}"] = 1.0

    # (7) Features 1,3,4 conjoined with the previous tag
    feats[f"W_i={word}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    feats[f"O_i={lower_word}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0
    feats[f"P_i={pos}+T_i-1={prev_tag}+T_i={curr_tag}"] = 1.0

    # (8) Length-k prefix for the current word, k = 1..4
    for k in range(1, 5):
        if len(word) >= k:
            prefix = word[:k]
            feats[f"PRE_i={prefix}+T_i={curr_tag}"] = 1.0

    # (9) Gazetteer membership
    gaz_in = is_in_gazetteer(word, curr_tag, gazetteers)
    feats[f"GAZ_i={gaz_in}+T_i={curr_tag}"] = 1.0

    # (10) Does current word start with a capital letter?
    starts_cap = word[0].isupper()
    feats[f"CAP_i={starts_cap}+T_i={curr_tag}"] = 1.0

    # (11) Position of current word (1-based)
    position = i + 1
    feats[f"POS_i={position}+T_i={curr_tag}"] = 1.0

    return feats

# -------------------------------------------------------------------------
# 3. Viterbi Decoding
# -------------------------------------------------------------------------

def viterbi_decode(sentence, tag_set, weights, gazetteers=None):
    """
    Runs Viterbi to get the best tag sequence for 'sentence'.
    """
    n = len(sentence)
    if n == 0:
        return []

    viterbi = {}
    backpointer = {}

    # Initialization for i=0
    for tag in tag_set:
        feats = extract_features(sentence, 0, "<START>", tag, gazetteers)
        score = sum(weights.get(f, 0.0) * val for f, val in feats.items())
        viterbi[(0, tag)] = score
        backpointer[(0, tag)] = None

    # Fill table
    for i in range(1, n):
        for tag in tag_set:
            best_score = float('-inf')
            best_prev_tag = None
            for prev_tag in tag_set:
                prev_score = viterbi[(i-1, prev_tag)]
                feats = extract_features(sentence, i, prev_tag, tag, gazetteers)
                score = prev_score + sum(weights.get(f, 0.0) * val for f, val in feats.items())
                if score > best_score:
                    best_score = score
                    best_prev_tag = prev_tag
            viterbi[(i, tag)] = best_score
            backpointer[(i, tag)] = best_prev_tag

    # Final step: find best tag at the last token
    best_final_score = float('-inf')
    best_final_tag = None
    for tag in tag_set:
        score = viterbi[(n - 1, tag)]
        if score > best_final_score:
            best_final_score = score
            best_final_tag = tag

    # Reconstruct path
    best_path = [best_final_tag]
    for i in range(n - 1, 0, -1):
        best_tag = best_path[-1]
        best_prev = backpointer[(i, best_tag)]
        best_path.append(best_prev)
    best_path.reverse()

    return best_path

# -------------------------------------------------------------------------
# 4. Structured Perceptron Training (SSGD)
# -------------------------------------------------------------------------

def compute_features_for_sequence(sentence, tag_seq, gazetteers, weights=None):
    """
    Aggregate feature vector for the entire sentence + tag_seq.
    Optionally compute its total score under 'weights'.
    """
    seq_features = defaultdict(float)
    total_score = 0.0
    prev_tag = "<START>"

    for i, _ in enumerate(sentence):
        curr_tag = tag_seq[i]
        feats = extract_features(sentence, i, prev_tag, curr_tag, gazetteers)
        for f, val in feats.items():
            seq_features[f] += val
            if weights is not None:
                total_score += weights.get(f, 0.0) * val
        prev_tag = curr_tag

    return seq_features, total_score

def structured_perceptron_train(train_data, dev_data, tag_set,
                                gazetteers=None, max_epochs=5, early_stop=True):
    """
    Train a linear model (weights) via structured perceptron using Viterbi + SSGD.
    No regularization, step size = 1.
    We do early stopping based on CoNLL F1 on the dev set.
    """
    weights = defaultdict(float)
    best_dev_f = -1.0
    best_weights = None

    def update_weights(weights, gold_feats, pred_feats):
        for f, val in gold_feats.items():
            weights[f] += val
        for f, val in pred_feats.items():
            weights[f] -= val

    for epoch in range(max_epochs):
        print(f"=== Epoch {epoch+1}/{max_epochs} ===")
        # Optionally shuffle train data
        # random.shuffle(train_data)

        # One pass over training data
        for sentence in train_data:
            gold_tags = [tok[3] for tok in sentence]
            pred_tags = viterbi_decode(sentence, tag_set, weights, gazetteers)
            if pred_tags != gold_tags:
                gold_f, _ = compute_features_for_sequence(sentence, gold_tags, gazetteers)
                pred_f, _ = compute_features_for_sequence(sentence, pred_tags, gazetteers)
                update_weights(weights, gold_f, pred_f)

        # Evaluate on dev using official conlleval-based function
        dev_p, dev_r, dev_f = evaluate_conll(dev_data, weights, tag_set, gazetteers)
        print(f"Dev set -> P={dev_p:.2f}, R={dev_r:.2f}, F1={dev_f:.2f}")

        if dev_f > best_dev_f:
            best_dev_f = dev_f
            best_weights = dict(weights)  # copy
        elif early_stop:
            print("No improvement in F1; stopping early.")
            break

    return best_weights if best_weights else dict(weights)

# -------------------------------------------------------------------------
# 5. CoNLL-based Evaluation
# -------------------------------------------------------------------------

def evaluate_conll(dataset, weights, tag_set, gazetteers=None):
    """
    Runs Viterbi over 'dataset', collects gold vs. predicted tags, and calls
    the 'evaluate' function from conlleval.py to get chunk-based precision/recall/F1.
    
    Returns (precision, recall, f1) as floats in [0..100].
    """
    gold_tags_all = []
    pred_tags_all = []
    for sentence in dataset:
        gold_tags = [item[3] for item in sentence]
        pred_tags = viterbi_decode(sentence, tag_set, weights, gazetteers)
        gold_tags_all.extend(gold_tags)
        pred_tags_all.extend(pred_tags)

    # The conlleval.evaluate function returns a tuple:
    # (precision%, recall%, f1%)
    # We'll call it with verbose=False to avoid printing details every time.
    p, r, f1 = evaluate(gold_tags_all, pred_tags_all, verbose=False)
    return p, r, f1

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ner_perceptron.py <train_file> <dev_file> <test_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    test_file = sys.argv[3]

    # 1) Read data
    train_data = read_conll_data(train_file)
    dev_data   = read_conll_data(dev_file)
    test_data  = read_conll_data(test_file)

    # 2) Collect tag set from training data
    all_tags = set()
    for sent in train_data:
        for (_, _, _, ne_tag) in sent:
            all_tags.add(ne_tag)
    tag_set = sorted(all_tags)  # e.g. [ "B-LOC", "B-MISC", ..., "O" ]

    # 3) (Optional) define gazetteers
    gazetteers = None  # or something like {'I-LOC': {"France", "Britain"}, ...}

    # 4) Train model
    print("Training structured perceptron...")
    learned_weights = structured_perceptron_train(
        train_data, dev_data, tag_set,
        gazetteers=gazetteers,
        max_epochs=10,
        early_stop=True
    )

    # 5) Evaluate on test set with official script
    test_p, test_r, test_f1 = evaluate_conll(test_data, learned_weights, tag_set, gazetteers)
    print(f"Test P={test_p:.2f}, R={test_r:.2f}, F1={test_f1:.2f}")

    # 6) Write predictions
    out_file = "predictions.txt"
    with open(out_file, 'w', encoding='utf-8') as outf:
        for sentence in test_data:
            pred_tags = viterbi_decode(sentence, tag_set, learned_weights, gazetteers)
            for (word, pos, chunk, gold_tag), p_tag in zip(sentence, pred_tags):
                outf.write(f"{word} {pos} {chunk} {gold_tag} {p_tag}\n")
            outf.write("\n")

    print(f"Wrote predictions to {out_file}")
