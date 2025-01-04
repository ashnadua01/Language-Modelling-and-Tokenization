## Assignment - 1

### Tokenization
- Filename: tokenizer.py
- Instructions: python3 tokenizer.py gives a prompt for sentence

The following entity replacements were implemented:
- URL
- MAILID
- DATE
- TIME
- AGE
- PERCENTAGE
- CURRENCY
- NUM
- MENTION
- HASHTAG

### N-Grams
- Filename: n_gram.py
- Instructions: python3 ngrams.py gives a prompt for a **tokenized** sentence

### Language Modelling
- Filename: language_model.py
- Instructions: python3 language_model.py <lm_type> <corpus_path> <n>
- n: n-gram model

Gives a prompt for a sentence, and prints the probability and perplexity of the given sentence.

The following are implemented:
- Good Turing Smoothing (method explained in report)
- Linear Interpolation (method explained in report)

Files with the perplexity of each sentence, are present in the **results** folder.

All models, are saved and loaded from the **checkpoints** folder.

### Generation
- Filename: generation.py
- Instructions: <type> <corpus_path> <k>
- Default value of n: 3
- Type: n for ngram model and i for interpolation model

Analysis:
- Corpus Influence: The “Ulysses” model predicts ‘by’ as the most likely next word, while the “Pride and Prejudice” model predicts ‘after’. This reflects the differences in the writing styles of the two books.
- Probability Distributions: The probabilities assigned to the top predictions are also different in the two models. This is due to differences in the frequency of these word sequences in the respective corpora.

### Perplexity Results
**LM1:** 
- Train: 389.19759689711213
- Test: 998.4044228768138

**LM2:** 
- Train: 57.43616835048108
- Test: 600.265796674905

**LM3:** 
- Train: 1023.2814008956045
- Test: 2826.796705996295

**LM4:** 
- Train: 234.23676003931897
- Test: 4357.013112635322