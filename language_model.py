from collections import defaultdict
from n_gram import NGram
import math
import numpy as np
from scipy.stats import linregress
import random
from tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import sys

class GoodTuringSmoothing:
    def __init__(self, ngrams, n):
        self.n = n
        self.ngrams_dict = ngrams
        self.ngram_counts = ngrams[n]
        self.vocabulary = ngrams[1]
        self.N = self.precompute_N()
        self.Z = self.precompute_Z()
        self.S, self.adjusted_freq = self.precompute_S()

    def precompute_N(self):
        N = defaultdict(int)
        for count in self.ngram_counts.values():
            N[count] += 1

        # for i, val in N.items():
        #     print(i, val)
        return N

    def precompute_Z(self):
        Z = {}
        r_values = sorted(list(self.N.keys()))
        
        for i, r in enumerate(r_values):
            if r == 1:
                q, t = 0, r_values[i+1]
                Z[r] = self.N[r] / (0.5 * (t - q))
            elif i == len(r_values) - 1:
                q, t = r_values[i-1], r
                Z[r] = self.N[r] / (t - q)
            else:
                q, t = r_values[i-1], r_values[i+1]
                Z[r] = self.N[r] / (0.5 * (t - q))
        
        return Z

    def precompute_S(self):
        S = {}
        log_r = np.log(list(self.Z.keys()))
        log_Z = np.log(list(self.Z.values()))
        slope, intercept, _, _, _ = linregress(log_r, log_Z)

        # print(slope, intercept)
        max_r = max(self.N.keys())
        
        for r in range(0, max_r+2):
            S[r] = np.exp(intercept + (np.log(r+1) * slope))

        adjusted_freq = {}
        for r in range(0, max_r+1):
            adjusted_freq[r] = (r+1) * S[r+1] / S[r]

        # plt.figure(figsize=(10, 6))
        # plt.scatter(log_r, log_Z, color='blue', label='Data')
        # plt.plot(log_r, intercept + slope * log_r, color='red', label='Fitted line')
        # plt.xlabel('log(r)')
        # plt.ylabel('log(Z)')
        # plt.title('Linear Regression Chart')
        # plt.legend()
        # plt.show()

        return S, adjusted_freq

    def smoothed_count(self, ngram):
        r = self.ngram_counts.get(ngram, 0)
        adjusted_count_ngram = self.adjusted_freq[r]
        n_1_gran = ngram[:-1]
        total_adjusted_count = 0

        for token in self.vocabulary.keys():
            ngram_possible = n_1_gran + token
            r_possible = self.ngram_counts.get(ngram_possible, 0)
            total_adjusted_count += self.adjusted_freq[r_possible]

        # extra UNK token
        total_adjusted_count += self.adjusted_freq[0]
        prob = adjusted_count_ngram / total_adjusted_count

        return prob

    def get_prob(self, ngram):
        prob = self.smoothed_count(ngram)
        return np.log(prob) if prob != 0 else float('-inf')
    
    def get_prob_sen(self, sentence):
        new_sen = tag_func(sentence, self.n-1)[0]
        len_sen = len(new_sen)

        ngram = NGram(self.n, [new_sen])
        ngram_sen = ngram.get_ngrams()

        log_prob_sum = 0
        for current_ngram in ngram.ngrams[self.n].keys():
            log_prob = self.get_prob(current_ngram)
            log_prob_sum += log_prob

        return log_prob_sum, len_sen
    
    def save(self):
        try:
            with open("./checkpoints/gts_model.pkl", "wb") as f:
                pickle.dump(self, f)
            return 1
        except Exception as e:
            print(f"Error saving model: {e}")
            return -1
        
    @classmethod
    def load(cls):
        try:
            with open("./checkpoints/gts_model.pkl", "rb") as f:
                loaded_model = pickle.load(f)
            return loaded_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
class PerplexityScore:
    def __init__(self, test_set, model, n):
        self.test_set = test_set
        self.model = model
        self.n = n

    def calculate_perplexity(self):
        perplex = 0
        num_sentences = len(self.test_set)
        perplexitities = []

        for sentence in self.test_set:
            tokenized_sen = Tokenizer(sentence).tokenize()
            sentence_log_prob_sum, len_sen = self.model.get_prob_sen(tokenized_sen)
            if not np.isnan(sentence_log_prob_sum) and not np.isinf(sentence_log_prob_sum):
                perplexity = np.exp(-sentence_log_prob_sum / len_sen)
                perplexitities.append((sentence, perplexity))
                perplex += perplexity
        
        if num_sentences == 0:
            return 0
        else:
            return perplexitities, perplex / num_sentences
    
def tag_func(corpus, n):
    new_corpus = []
    for sen in corpus:
        curr = []
        for l in range(n):
            curr.append('<s>')

        for l in sen:
            curr.append(l)

        curr.append('</s>')
        new_corpus.append(curr)
    
    return new_corpus

class LinearInterpolation:
    def __init__(self, tokenized_corpus, n):
        self.n = n
        
        self.final_tokenized_ngram1 = tag_func(tokenized_corpus, n-1) #changed here
        
        self.ngram = NGram(self.n, tokenized_corpus)
        self.nsem = self.ngram.get_ngrams()

        self.ngram3 = self.ngram.ngrams[3]
        self.ngram2 = self.ngram.ngrams[2]
        self.ngram1 = self.ngram.ngrams[1]

        self.lambda_values = self.calculate_lambda()

    def calculate_lambda(self):
        l1 = l2 = l3 = 0

        N = sum(self.ngram1.values())
        for ngram, count in self.ngram3.items():
            if count <= 0:
                continue
            t1_t2 = ngram[:-1]
            t2_t3 = ngram[1:]
            t3 = ngram[-1:]
            t2 = ngram[1:2]

            case1 = case2 = case3 = 0
            if self.ngram2[t1_t2] > 1:
                case1 = (self.ngram3[ngram] - 1) / (self.ngram2[t1_t2] - 1)
            if self.ngram1[t2] > 1:
                case2 = (self.ngram2[t2_t3] - 1) / (self.ngram1[t2] - 1)
            if N > 1:
                case3 = (self.ngram1[t3] - 1) / (N - 1)

            max_case = max(case1, max(case2, case3))
            if max_case == case1:
                l3 += count
            elif max_case == case2:
                l2 += count
            else:
                l1 += count

        total = l1 + l2 + l3
        l1 = l1 / total
        l2 = l2 / total
        l3 = l3 / total

        # print(l1, l2, l3)
        return [l1, l2, l3]
    
    def get_prob(self, ngram):
        prob = 0
        epsilon = 1e-10

        # unigram
        if len(ngram) == 1:
            unigram = ngram if ngram in self.ngram1 or self.ngram1[ngram] != 0 else "UNK"
            if unigram != "UNK":
                count = self.ngram1[unigram]
                if count == 0:
                    prob = epsilon
                else:
                    total = sum(list(self.ngram1.values()))
                    prob = count / total
            else:
                prob = epsilon

        # # bigram
        elif len(ngram) == 2:
            bigram = ngram if ngram in self.ngram2 else "UNK"
            if bigram != "UNK":
                count = self.ngram2[bigram]
                context = tuple(ngram[:-1])

                if context in self.ngram1:
                    if self.ngram1[context] != 0:
                        total = self.ngram1[context]
                        prob = count / total
                    else:
                        prob = epsilon
                else:
                    prob = epsilon
            else:
                prob = epsilon

        # # trigram
        elif len(ngram) == 3:
            trigram = ngram if ngram in self.ngram3 else "UNK"
            if trigram != "UNK":
                count = self.ngram3[trigram]
                context = ngram[:-1]
                if context in self.ngram2:
                    if self.ngram2[context] != 0:
                        total = self.ngram2[context]
                        prob = count / total
                    else:
                        prob = epsilon
                else:
                    prob = epsilon
            else:
                prob = epsilon

        return prob
    
    def get_prob_sen(self, new_sen):
        total_prob = 0
        new_sen = new_sen[0]

        for i in range(2, len(new_sen)):
            unigram = tuple([new_sen[i]])
            bigram = tuple(new_sen[i-1:i+1])
            trigram = tuple(new_sen[i-2:i+1])

            prob_unigram = self.get_prob(unigram)
            prob_bigram = self.get_prob(bigram)
            prob_trigram = self.get_prob(trigram)

            prob = self.lambda_values[0] * prob_unigram + self.lambda_values[1] * prob_bigram + self.lambda_values[2] * prob_trigram
            total_prob += np.log(prob)

        return total_prob, len(new_sen)
    
    def save(self):
        try:
            with open("./checkpoints/li_model.pkl", "wb") as f:
                pickle.dump(self, f)
            return 1
        except Exception as e:
            print(f"Error saving model: {e}")
            return -1
        
    @classmethod
    def load(cls):
        try:
            with open("./checkpoints/li_model.pkl", "rb") as f:
                loaded_model = pickle.load(f)
            return loaded_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
class PerplexityScoreLinearInterpolation:
    def __init__(self, test_set, model, n):
        self.test_set = test_set
        self.model = model
        self.n = n

    def calculate_perplexity(self):
        perplex = 0
        num_sentences = len(self.test_set)
        perplexitities = []

        for sentence in self.test_set:
            tokenized_sen = Tokenizer(sentence).tokenize()
            new_sen = tag_func(tokenized_sen, self.n-1)

            sentence_log_prob_sum, len_sen = self.model.get_prob_sen(new_sen)
            if not np.isnan(sentence_log_prob_sum) and not np.isinf(sentence_log_prob_sum):
                perplexity = np.exp(-sentence_log_prob_sum / len_sen)
                perplexitities.append((sentence, perplexity))
                perplex += perplexity
                # print(perplexity)
            
        if num_sentences == 0:
            return 0
        else:
            return perplexitities, perplex / num_sentences

def read_corpus(corpus_path):
    with open(corpus_path, 'r') as file:
        data = file.read()
    return data
        
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 language_model.py <lm_type> <corpus_path> <n>")
        sys.exit(1)

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    n = int(sys.argv[3])

    corpus = read_corpus(corpus_path)
    tokenizer = Tokenizer(corpus)

    sentences = tokenizer.tokenize_sentence(tokenizer.corpus)
    train_corpus, test_corpus = train_test_split(sentences, test_size=1000, random_state=42)
    train_corpus_for_tokenize = " ".join(train_corpus)

    tokenizer = Tokenizer(train_corpus_for_tokenize)
    tokenized_train = tokenizer.tokenize()
    final_tokenized = tag_func(tokenized_train, n-1)

    ng = NGram(n, final_tokenized)
    ngram = ng.get_ngrams()

    if lm_type == "g":
        model = GoodTuringSmoothing(ng.ngrams, n)

    elif lm_type == "i":
        model = LinearInterpolation(tokenized_train, n)

    inp = input("input sentence:\n")

    tokenized_sen = Tokenizer(inp).tokenize()
    new_sen = tag_func(tokenized_sen, n-1)

    sentence_log_prob_sum, len_sen = model.get_prob_sen(new_sen)
    if not np.isnan(sentence_log_prob_sum) and not np.isinf(sentence_log_prob_sum):
        perplexity = np.exp(-sentence_log_prob_sum / len_sen)
    
    print("score: ", math.exp(sentence_log_prob_sum))
    print("perplexity: ", perplexity)

# n1 = "/Users/ashnadua/Desktop/inlp-Assignment1/Pride and Prejudice - Jane Austen.txt"
# n2 = "/Users/ashnadua/Downloads/OneDrive_1_16-1-2024/Ulysses - James Joyce.txt"

# corpus1 = read_corpus(n1)
# corpus2 = read_corpus(n2)

# n = 3

# tokenizer1 = Tokenizer(corpus1)
# tokenizer2 = Tokenizer(corpus2)

# sentences1 = tokenizer1.tokenize_sentence(tokenizer1.corpus)
# train_corpus1, test_corpus1 = train_test_split(sentences1, test_size=1000, random_state=42)
# train_corpus_for_tokenize1 = " ".join(train_corpus1)

# tokenizern1 = Tokenizer(train_corpus_for_tokenize1)
# tokenized_train1 = tokenizern1.tokenize()
# final_tokenized1 = tag_func(tokenized_train1, n-1)

# ng1 = NGram(n, final_tokenized1)
# ngram = ng1.get_ngrams()

# sentences2 = tokenizer2.tokenize_sentence(tokenizer2.corpus)
# train_corpus2, test_corpus2 = train_test_split(sentences2, test_size=1000, random_state=42)
# train_corpus_for_tokenize2 = " ".join(train_corpus2)

# tokenizern2 = Tokenizer(train_corpus_for_tokenize2)
# tokenized_train2 = tokenizern2.tokenize()
# final_tokenized2 = tag_func(tokenized_train2, n-1)

# ng2 = NGram(n, final_tokenized2)
# ngram = ng2.get_ngrams()

# lm1 = GoodTuringSmoothing(ng1.ngrams, 3)
# lm3 = GoodTuringSmoothing(ng2.ngrams, 3)

# lm2 = LinearInterpolation(tokenized_train1, 3)
# # lm4 = LinearInterpolation(tokenized_train2, 3)

# perplexity_calc1 = PerplexityScore(train_corpus1, lm1, 3)
# perplexitities, avg = perplexity_calc1.calculate_perplexity()

# with open(f"{2021101072}_LM1_train-perplexity.txt", "w") as f:
#         f.write(f"{avg}\n")
#         for sentence, perplexity in perplexitities:
#             f.write(f"{sentence}\t{perplexity}\n")

# perplexity_calc2 = PerplexityScore(test_corpus1, lm1, 3)
# perplexitities, avg = perplexity_calc2.calculate_perplexity()

# with open(f"{2021101072}_LM1_test-perplexity.txt", "w") as f:
#         f.write(f"{avg}\n")
#         for sentence, perplexity in perplexitities:
#             f.write(f"{sentence}\t{perplexity}\n")

# perplexity_calc1 = PerplexityScore(train_corpus2, lm3, 3)
# perplexitities, avg = perplexity_calc1.calculate_perplexity()

# with open(f"{2021101072}_LM3_train-perplexity.txt", "w") as f:
#         f.write(f"{avg}\n")
#         for sentence, perplexity in perplexitities:
#             f.write(f"{sentence}\t{perplexity}\n")

# perplexity_calc2 = PerplexityScore(test_corpus2, lm3, 3)
# perplexitities, avg = perplexity_calc2.calculate_perplexity()

# with open(f"{2021101072}_LM3_test-perplexity.txt", "w") as f:
#         f.write(f"{avg}\n")
#         for sentence, perplexity in perplexitities:
#             f.write(f"{sentence}\t{perplexity}\n")

# perplexity_calc1 = PerplexityScoreLinearInterpolation(train_corpus1, lm2, 3)
# perplexitities, avg = perplexity_calc1.calculate_perplexity()

# with open(f"{2021101072}_LM2_train-perplexity.txt", "w") as f:
#         f.write(f"{avg}\n")
#         for sentence, perplexity in perplexitities:
#             f.write(f"{sentence}\t{perplexity}\n")

# # perplexity_calc2 = PerplexityScoreLinearInterpolation(test_corpus1, lm2, 3)
# # perplexitities, avg = perplexity_calc2.calculate_perplexity()

# # with open(f"{2021101072}_LM2_test-perplexity.txt", "w") as f:
# #         f.write(f"{avg}\n")
# #         for sentence, perplexity in perplexitities:
# #             f.write(f"{sentence}\t{perplexity}\n")

# # perplexity_calc1 = PerplexityScoreLinearInterpolation(train_corpus2, lm4, 3)
# # perplexitities, avg = perplexity_calc1.calculate_perplexity()

# # with open(f"{2021101072}_LM4_train-perplexity.txt", "w") as f:
# #         f.write(f"{avg}\n")
# #         for sentence, perplexity in perplexitities:
# #             f.write(f"{sentence}\t{perplexity}\n")

# # perplexity_calc2 = PerplexityScoreLinearInterpolation(test_corpus2, lm4, 3)
# # perplexitities, avg = perplexity_calc2.calculate_perplexity()

# # with open(f"{2021101072}_LM4_test-perplexity.txt", "w") as f:
# #         f.write(f"{avg}\n")
# #         for sentence, perplexity in perplexitities:
# #             f.write(f"{sentence}\t{perplexity}\n")