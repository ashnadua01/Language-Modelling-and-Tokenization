from n_gram import NGram
import numpy as np
import random
import sys
from collections import defaultdict
from tokenizer import Tokenizer
from language_model import LinearInterpolation

class Generator:
    def __init__(self, model, k, n):
        self.model = model
        self.k = k
        self.n = n

    def generate_sequence_ngrams(self, context):
        if(len(context[0]) < self.n):
            self.n = len(context[0])
        
        sequence = context[0]
        context_ngram = tuple(sequence[-self.n:])
        print(context_ngram)

        count = self.model[self.n].get(context_ngram, 0)

        vocabulary = self.model[1]
        probs = []
        
        for word, counter in vocabulary.items():
            if count != 0 and self.model[self.n+1].get(context_ngram + word, 0) != 0:
                probs.append((word, self.model[self.n+1][context_ngram + word] / count))
            else:
                probs.append((word, 0))

        if all(prob == 0 for _, prob in probs):
            for i in range(self.n-1, 0, -1):
                context_ngram = tuple(sequence[-i:])
                count = self.model[i].get(context_ngram, 0)

                if count != 0:
                    for word, counter in vocabulary.items():
                        if self.model[i+1].get(context_ngram + word, 0) != 0:
                            probs.append((word, self.model[i+1][context_ngram + word] / count))

                if any(prob > 0 for _, prob in probs):
                    print(f"Back Off probability from {i}-gram model")
                    break

        if not any(prob > 0 for _, prob in probs):
            probs.append(("</s>", 1e-10))

        probs.sort(key=lambda x: x[1], reverse=True)
        for word, prob in probs[:self.k]:
            if prob != 0:
                print(f"{word} {prob}")

    def generate_sequence_linear_interpolation(self, context):
        if(len(context[0]) < self.n):
            self.n = len(context[0])

        sequence = context[0]
        context_ngram = tuple(sequence[-self.n:])

        vocabulary = self.model.ngram1
        probs = []

        for word in vocabulary.keys():
            unigram = word
            bigram = context_ngram[-1:] + word
            trigram = context_ngram[-2:] + word

            prob_unigram = self.model.get_prob(unigram)
            prob_bigram = self.model.get_prob(bigram)
            prob_trigram = self.model.get_prob(trigram)

            prob = self.model.lambda_values[0] * prob_unigram + self.model.lambda_values[1] * prob_bigram + self.model.lambda_values[2] * prob_trigram
            probs.append((word, prob))

        probs.sort(key=lambda x: x[1], reverse=True)        
        for word, prob in probs[:self.k]:
                if prob != 0:
                    print(f"{word} {prob}")

def read_corpus(corpus_path):
    with open(corpus_path, 'r') as file:
        data = file.read()
    return data

# N = [3]
# corpus_path = "/Users/ashnadua/Desktop/2021101072_assignment1/Pride and Prejudice - Jane Austen.txt"
# corpus = read_corpus(corpus_path)
# tokenizer = Tokenizer(corpus)
# tokenized_corpus = tokenizer.tokenize()

# sen = input("input sentence:\n")
# tokenizer = Tokenizer(sen)
# tokenized_sen = tokenizer.tokenize()

# k = 3

# for n in N:
#     ng = NGram(n+1, tokenized_corpus)
#     ngram = ng.get_ngrams()

#     model = LinearInterpolation(tokenized_corpus, n)
#     print(n)
#     gen = Generator(model, k, n)
#     print("output:")
#     gen.generate_sequence_linear_interpolation(tokenized_sen)
                    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 generator.py <type> <corpus_path> <k>")
        sys.exit(1)
    
    print("Default value of n = 3")
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])
    n = 3

    corpus = read_corpus(corpus_path)
    tokenizer = Tokenizer(corpus)
    tokenized_corpus = tokenizer.tokenize()

    inp = input("input sentence:\n")
    tokenizer = Tokenizer(inp)
    tokenized_sen = tokenizer.tokenize()

    if(len(tokenized_sen[0]) < 2 and lm_type == "i"):
        print("Interpolation generation not possible")
        exit

    ng = NGram(n+1, tokenized_corpus)
    ngram = ng.get_ngrams()

    if lm_type == "n":
        model = ng.ngrams
        gen = Generator(model, k, n)
        print("output:")
        gen.generate_sequence_ngrams(tokenized_sen)

    elif lm_type == "i":
        model = LinearInterpolation(tokenized_corpus, n)
        gen = Generator(model, k, n)
        print("output:")
        gen.generate_sequence_linear_interpolation(tokenized_sen)