'''BPE (Byte Pair Encoding) is a subword tokenization algorithm that starts with a base vocabulary of individual characters and iteratively merges the most frequent pairs of tokens to create new tokens.
One approach is to treat the words a sequence of Unicode characters, and then learn which pairs of characters or subwords are most common in the training data.
This approach, however, faces the issue of characters that fall outside of the training corpus, such as emojis or rare characters.
To address this, BPE can be applied at the byte level, where the text is first encoded into bytes (using UTF-8 encoding) and then the BPE algorithm is applied to these byte sequences.
For example, the hanzi character "好" would be represented as the byte sequence [0xE5, 0xA5, 0xBD] in UTF-8 encoding. BPE would then learn to merge these byte pairs into subword tokens, 
allowing it to effectively handle a wide range of characters, including those that are not present in the training data. This makes BPE a powerful and flexible tokenization method for various languages and applications.'''

# We start by training a BPE tokenizer on a small corpus of characteers.
# Step 1: Calculate the frequency of each character in the training data, and create a vocabulary of individual characters.
# Step 2: Identify the most frequent pair of tokens in the training data, and merge them into a new token. This new token is added to the vocabulary, and the frequency of the merged pair is updated.
# Step 3: Repeat step 2 until a predefined vocabulary size is reached or there are no more pairs to merge.

"Let's implement the BPE from scratch."

# First, create a small corpus of text data to train the BPE tokenizer on.

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# Next, we pre-tokenize the text data into words.

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs) # frequency of each word (pretokenized) in the corpus

# Next, compute the base vocabulary of individual characters.

alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)

vocab = ["<|endoftext|>"] + alphabet.copy() # We add the special end of text token to the vocabulary.

# We now need to split each word into individual characters, to be able to start training:

splits = {word: [c for c in word] for word in word_freqs.keys()}

'''splits = {
    "low":   ["l", "o", "w"],
    "lower": ["l", "o", "w", "e", "r"]
}'''

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq # since this word also appears "freq" times
    return pair_freqs # We compute the frequency of each pair of tokens in the splits with two characters.

pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

# We can find the most frequent pair

best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)

merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")

# Then, we have to apply that merge to the splits:

def merge_pair(a, b, splits):
    for word in word_freqs.keys():
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits

# Now we have all the components to train the BPE tokenizer.

vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq # find the most frequent pair
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])


# To tokenize a new text, we pre-tokenize it, split it, then apply all the merge rules learned:

def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])