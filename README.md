# Quora Duplicate Question Detection: NLP Feature Evolution

## 1. Project Overview (The "What" & "Why")

### What is this project?
This project focuses on Natural Language Processing (NLP) to identify duplicate questions within a 90,000-row sample of the Quora dataset[cite: 2]. It traces the evolution of text representation and feature engineering—from basic text cleaning and Bag-of-Words (BOW) representations to advanced fuzzy-matching heuristics and semantic Word2Vec embeddings. All preprocessing and feature extraction workflows are documented in `BOW_Preprocessing_and_Adv_Features.ipynb`[cite: 2].

### Why was this built? (The Use Cases)
1. **The Business Use Case:** Platforms like Quora suffer from fragmented knowledge when users ask the exact same question in slightly different ways (e.g., "How do I invest with little money?" vs. "I am quite poor and I want to be very rich. What should I do?")[cite: 2]. Detecting semantic duplicates allows platforms to merge threads, consolidating expert answers and drastically improving the user search experience.
2. **The Engineering Use Case:** Raw text is messy and cannot be fed directly into machine learning models. This project serves as a comprehensive pipeline for text normalization, manual feature extraction (syntactic similarity), and dense vectorization (semantic similarity), evaluating the strengths and computational trade-offs of each approach.

---

## 2. Text Preprocessing Strategy (The "How")

Text data requires strict sanitization to ensure the machine learning models focus on meaningful vocabulary rather than formatting noise. The following preprocessing steps were systematically applied to every question[cite: 2]:

* **Standardization:** Lowercasing all text and replacing mathematical/currency symbols (%, \$, €, ₹, @) with their string equivalents (e.g., "percent", "dollar") to preserve their financial or mathematical context[cite: 2].
* **Magnitude Normalization:** Replacing large numeric strings with shorthand (e.g., replacing '000000000' with 'b' for billions, '000000' with 'm' for millions)[cite: 2].
* **Decontraction:** Expanding English contractions using a comprehensive dictionary mapping (e.g., converting "couldn't've" to "could not have") to unify word representations[cite: 2].
* **Noise Removal:** Stripping out HTML tags using `BeautifulSoup` and utilizing regular expressions to remove all non-alphanumeric punctuation[cite: 2].

---

## 3. Advanced Feature Engineering 

Before relying on heavy deep-learning embeddings, we extracted critical heuristic features to capture the structural and syntactic similarities between question pairs:

* **Basic Length & Word Share:** Extracted raw word counts (`q1_num_words`, `q2_num_words`), total unique words, and the ratio of shared words (`word_share`)[cite: 2].
* **Token Features (NLTK):** Utilized the `nltk` library to isolate standard words from "stopwords." We engineered ratio features comparing common words (`cwc_min`/`max`), common stopwords (`csc_min`/`max`), and boolean flags indicating if the first or last words of the questions were identical[cite: 2].
* **Distance & Substring Features:** Used the `distance` library to calculate the absolute length difference and the longest common substring ratio between questions[cite: 2].
* **Fuzzy Matching:** Leveraged `thefuzz` library to calculate Levenshtein distance metrics, generating powerful syntactic features: `fuzz_ratio`, `fuzz_partial_ratio`, `token_sort_ratio`, and `token_set_ratio`[cite: 2].

---

## 4. Text Representation & Modeling Evolution

To feed this data into a predictive model, the raw text was mathematically encoded in progressive phases of complexity:

### Phase 1: Bag of Words (BOW)
* **What it is:** Represents text by counting the frequency of words, completely ignoring order and context.
* **Why we used it:** It provides a highly interpretable, lightweight baseline. If two questions share a massive amount of exact vocabulary, BOW captures this easily.
* **The Limitation:** BOW suffers from massive sparsity and is entirely "context-blind." It cannot recognize that "poor" and "broke" mean the same thing.

### Phase 2: Advanced Heuristic Features
* **What it is:** The mathematical syntactic features extracted in Section 3 (Fuzzy ratios, token matching, substring overlap)[cite: 2].
* **Why we used it:** To bridge the gap left by BOW. Fuzzy matching catches typos, while token-sort ratios catch questions that have the same words in a different order.
* **The Limitation:** These features are computationally heavy to extract across millions of rows and still do not inherently understand the *definition* of the words.

### Phase 3: Semantic Word2Vec Embeddings
* **What it is:** A neural network-based approach that maps words to dense, 100-dimensional vectors based on their contextual neighborhood. 
* **Why we used it:** Word2Vec understands semantic relationships. By averaging the word vectors for each question and calculating the **Cosine Similarity**, the model can identify that "investing with no money" and "getting rich while poor" share semantic proximity, even if their exact word overlap is minimal.
* **The Improvement:** Drastically reduces dimensionality compared to BOW while capturing the actual meaning and intent behind the questions.

---

## 5. Core Limitations & Bottlenecks

1. **The Averaging Flaw:** By taking the mathematical *mean* of all Word2Vec word embeddings to create a single "sentence vector," the model loses the sequential order of the words. "The dog bit the man" and "The man bit the dog" result in the exact same average vector, despite having vastly different meanings.
2. **Out-of-Vocabulary (OOV) Words:** Word2Vec can only generate vectors for words it saw during training. Rare words, modern slang, or severe misspellings result in zero-vectors, dropping valuable information.

## 6. Strategic Next Steps

To push the predictive accuracy past the limits of averaged Word2Vec and BOW, the next architectural iterations will involve:
* **TF-IDF Weighted Word2Vec:** Instead of a simple average, using TF-IDF scores to give more mathematical weight to rare, important words (like "invest") and less weight to common words when building the sentence vector.
* **Transformer Architectures (BERT):** Migrating to state-of-the-art transformer models (like HuggingFace's `SentenceTransformers`) which calculate embeddings based on the *entire* sentence context simultaneously, completely resolving the sequential order limitations of averaged Word2Vec.
