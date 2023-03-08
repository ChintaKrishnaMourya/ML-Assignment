from typing import List
from nltk import word_tokenize # the nltk word tokenizer
from spacy.lang.en import English  # for the spacy tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math
from collections import Counter
import nltk

nltk.download('punkt')
nlp = English()

def load_corpus(filename: str) -> List[str]:
    with open(filename) as f:
        corpus = f.readlines()
    return corpus

def nltk_tokenize(sentence: str) -> List[str]:
    return word_tokenize(sentence)

def spacy_tokenize(sentence: str) -> List[str]:
    doc = nlp(sentence)
    return [token.orth_ for token in doc]

def tokenize(sentence: str) -> List[str]:
    return nltk_tokenize(sentence)

# part of solution to 2a
def count_bigrams(corpus):
    bigrams = {}
    for sentence in corpus:
        tokens = tokenize(sentence)
        for i in range(len(tokens) - 1):
            bigram = tokens[i]+" "+tokens[i + 1]
            bigrams[bigram]=bigrams.get(bigram,0)+1
    return bigrams

# part of solution to 2a
def count_trigrams(corpus):
    trigrams = {}
    for sentence in corpus:
        tokens = tokenize(sentence)
        for i in range(len(tokens) - 2):
            trigram = tokens[i] +" "+tokens[i + 1]+" "+tokens[i + 2]
            trigrams[trigram]=trigrams.get(trigram,0)+1
    return trigrams

# part of solution to 2b
def bigram_frequency(bigram:str, bigram_freq_dict):
  return bigram_freq_dict.get(bigram,0)

# part of solution to 2c
def trigram_frequency(trigram:str, trigram_freq_dict):
  return trigram_freq_dict.get(trigram,0)

# part of solution to 2d
def get_total_frequency(ngram_freq_dict):
    return sum(ngram_freq_dict.values())

# part of solution to 2e
def get_probability(ngram, ngram_freq_dict):
    ngram_count = ngram_freq_dict.get(ngram, 0)
    return ngram_count / get_total_frequency(ngram_freq_dict)

# part of solution to 3a
def forward_transition_probability(seq_of_three_tokens: list, bigram_counts: dict, trigram_counts: dict) -> list:
    fw_prob = []
    bigram = " ".join(seq_of_three_tokens[:2])
    trigram = " ".join(seq_of_three_tokens)
    bigram_count = bigram_counts.get(bigram, 0)
    trigram_count = trigram_counts.get(trigram, 0)
    if bigram_count > 0:
        fw_prob.append(trigram_count / bigram_count)
    else:
        fw_prob.append(0)
    return fw_prob


# part of solution to 3b
def backward_transition_probability(seq_of_three_tokens: list, bigram_counts: dict, trigram_counts: dict) -> list:
    bw_prob = []
    bigram = " ".join(seq_of_three_tokens[1:])
    trigram = " ".join(seq_of_three_tokens)
    bigram_count = bigram_counts.get(bigram, 0)
    trigram_count = trigram_counts.get(trigram, 0)
    if bigram_count > 0:
        bw_prob.append(trigram_count / bigram_count)
    else:
        bw_prob.append(0)
    return bw_prob


# part of solution to 3c
def compare_fw_bw_probability(fw_prob: float, bw_prob: float) -> bool:
    equivalence_test = fw_prob == bw_prob
    return equivalence_test


# part of solution to 3d
def sentence_likelihood(sentence: str, bigram_counts: dict, trigram_counts: dict) -> float:
    tokens = tokenize(sentence)
    likelihoods = []
    for i in range(2, len(tokens)):
        trigram = tokens[i-2:i+1]
        fw_prob = forward_transition_probability(trigram, bigram_counts, trigram_counts)[0]
        likelihoods.append(fw_prob)
    log_likelihood = sum([math.log(p) for p in likelihoods if p])
    return log_likelihood

# 4a
def neural_tokenize(sentence: str):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer_output = tokenizer(
        sentence, return_tensors="pt"
    )
    return tokenizer_output


# 4b
def neural_logits(tokenizer_output):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    logits = model(**tokenizer_output).logits
    return logits


# 4c
def normalize_probability(logits):
    softmax = torch.nn.Softmax(dim=2)
    softmax_logits = softmax(logits)
    return softmax_logits


# 4d.i
def neural_fw_probability(softmax_logits, tokenizer_output):
    input_ids = tokenizer_output["input_ids"]
    probabilities = softmax_logits[0, :, input_ids[0]].diag()
    return probabilities


# 4d.ii
def neural_likelihood(diagonal_of_probs):
    log_probs = torch.log(diagonal_of_probs)
    likelihood = log_probs.sum()
    return likelihood