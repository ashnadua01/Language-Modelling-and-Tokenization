from collections import defaultdict, Counter
from tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np

class NGram:
    def __init__(self, n, tokenized_corpus):
        self.n = int(n)
        self.ngrams = {i: defaultdict(int) for i in range(1, n+1)}
        self.tokenized_corpus = tokenized_corpus

    def get_ngrams(self):
        for sentence in self.tokenized_corpus:
            for i in range(1, self.n+1):
                for j in range(len(sentence) - i + 1):
                    ngram = tuple(sentence[j:j+i])
                    self.ngrams[i][ngram] += 1

        self.model = {}
        for ngram, count in self.ngrams[i].items():
            if self.n == 1:
                total = sum(list(self.ngrams[self.n].values()))
            else:
                context = ngram[:-1]
                total = self.ngrams[self.n-1][context]
            self.model[ngram] = count / total

# n = int(input("Enter n\n"))
# corpus = input("Enter corpus path\n")

# def read_corpus(corpus_path):
#     with open(corpus_path, 'r') as file:
#         data = file.read()
#     return data

# corpus = read_corpus(corpus)
# tokenizer = Tokenizer(corpus)
# tokenized_text = tokenizer.tokenize()

# final_tokenized = []
# for sen in tokenized_text:
#     curr = []

#     for l in range(n-1):
#         curr.append('<s>')
    
#     for l in sen:
#         curr.append(l)
    
#     curr.append('</s>')
#     final_tokenized.append(curr)

# print(final_tokenized)

# ng = NGram(n, final_tokenized)
# ngrams = ng.get_ngrams()

# for n_gram, probability in ngrams[n].items():
#     print(f"The probability of the n-gram '{n_gram}' is {probability}")
