
# 📘 NLP Fundamentals — Course Notes

A comprehensive guide covering foundational Natural Language Processing (NLP) concepts, text preprocessing pipelines, feature engineering, and model evaluation techniques.

---

## 📑 Table of Contents

1. [Introduction to NLP & Applications](#1-introduction-to-nlp--applications)
2. [Challenges in NLP](#2-challenges-in-nlp)
3. [The NLP Pipeline & Workflow](#3-the-nlp-pipeline--workflow)
4. [Text Data and Datasets](#4-text-data-and-datasets)
5. [Text Cleaning and Normalization](#5-text-cleaning-and-normalization)
6. [Tokenization](#6-tokenization)
7. [Stop Word Removal](#7-stop-word-removal)
8. [Stemming and Lemmatization](#8-stemming-and-lemmatization)
9. [Regular Expressions (Regex)](#9-regular-expressions-regex)
10. [Word Frequency and Vocabulary](#10-word-frequency-and-vocabulary)
11. [N-grams: Unigrams, Bigrams, Trigrams](#11-n-grams-unigrams-bigrams-trigrams)
12. [Bag of Words (BoW)](#12-bag-of-words-bow)
13. [CountVectorizer](#13-countvectorizer)
14. [TF-IDF (Term Frequency – Inverse Document Frequency)](#14-tf-idf-term-frequency--inverse-document-frequency)
15. [CountVectorizer vs. TfidfVectorizer](#15-countvectorizer-vs-tfidfvectorizer)
16. [Train-Test Split for Text Data](#16-train-test-split-for-text-data)

---

## 1. Introduction to NLP & Applications

**Natural Language Processing (NLP)** is a branch of Artificial Intelligence (AI) that enables computers to understand, interpret, and generate human language.


```

Human Language (Text / Speech) ───► [ NLP Engine ] ───► Useful Output (Label, Summary, Action)

```

### The Two Sides of NLP

| Branch | Definition | Example Tasks |
| :--- | :--- | :--- |
| **NLU** *(Natural Language Understanding)* | Reading text and extracting meaning, intent, or sentiment | Review classification, spam detection |
| **NLG** *(Natural Language Generation)* | Producing coherent, human-like text | Article summarization, chatbot dialogue generation |

### Why NLP is Essential
* Most real-world data is **unstructured text** (emails, customer reviews, support tickets, tweets).
* Machine learning algorithms cannot process raw strings—they require numerical matrices.
* NLP serves as the bridge: **Text ──► Numerical Vectors ──► Insights / Predictions**.

### Real-World Applications
* **Machine Translation:** Google Translate, DeepL (e.g., Hindi ↔ English)
* **Conversational Agents:** Siri, Alexa, ChatGPT, customer support bots
* **Sentiment Analysis:** Classifying feedback as positive, neutral, or negative
* **Spam Filtering:** Classifying inbound emails as legitimate or junk
* **Information Retrieval:** Semantic search queries on Google, Bing
* **Text Summarization:** Condensing long articles into concise overviews

### Evolution of NLP
1. **Rule-Based Systems (1950s–1980s):** Hand-crafted grammar rules, regular expressions, and curated dictionaries.
2. **Statistical & Machine Learning (1990s–2010s):** Feature extraction (BoW, TF-IDF) paired with models like Naive Bayes, SVM, and Logistic Regression.
3. **Deep Learning (2013+):** Dense word embeddings (Word2Vec, GloVe), RNNs, GRUs, and LSTMs.
4. **Transformers & LLMs (2017+):** Self-attention mechanisms, BERT, GPT, and modern Large Language Models.

---

## 2. Challenges in NLP

Human language is context-dependent, nuanced, and ambiguous.

| Challenge | Definition | Example |
| :--- | :--- | :--- |
| **Lexical Ambiguity** | A single word has multiple distinct meanings. | *"bank"* (river bank vs. financial institution) |
| **Syntactic Ambiguity** | A sentence can be parsed with multiple structural interpretations. | *"I saw the man with the telescope."* (Who had the telescope?) |
| **Sarcasm & Irony** | The literal meaning is the opposite of the intended sentiment. | *"Great service! Waited 2 hours for cold food."* |
| **Context Dependency** | Pronouns or words depend on previous sentences. | *"He dropped it."* (Requires antecedent resolution) |
| **Synonymy** | Multiple different words convey the exact same concept. | *buy*, *purchase*, *acquire*, *procure* |
| **Slang & Typos** | Informal text, abbreviations, and non-dictionary words. | *"gr8"*, *"plz"*, *"u"*, *"luv"* |
| **Code-Mixing** | Alternating between two or more languages in one sentence. | *"Yeh movie bahut awesome thi."* (Hindi + English) |
| **Idiomatic Expressions** | Non-literal, figurative meaning. | *"Kick the bucket"* (to die) |
| **Domain-Specific Jargon** | Words that shift meaning depending on the field. | *"cell"* (biology vs. telecommunications vs. spreadsheets) |

---

## 3. The NLP Pipeline & Workflow

Every standard NLP task follows a systematic lifecycle:


```

1. Collect ──► 2. Clean ──► 3. Preprocess ──► 4. Explore (EDA) ──► 5. Feature Extraction
│
2. Deploy  ◄── 7. Evaluate ◄────────────── 6. Train-Test Split ◄────────────┘

```

1. **Data Collection:** Gather text from web scraping, public datasets, or APIs.
2. **Text Cleaning:** Strip HTML, remove URLs, strip special characters/punctuation, lowercase.
3. **Preprocessing:** Tokenization, stop word filtering, stemming or lemmatization.
4. **Exploratory Data Analysis (EDA):** Analyze token distributions, vocabulary sizes, and common phrases.
5. **Feature Extraction:** Transform text into numerical feature matrices (BoW, TF-IDF).
6. **Train-Test Split:** Partition data into independent training and validation subsets.
7. **Model Training:** Train supervised/unsupervised machine learning models.
8. **Evaluation:** Measure performance using Accuracy, Precision, Recall, and F1-score.
9. **Deployment:** Serve the model as an API endpoint or embedded service.

---

## 4. Text Data and Datasets

### Core Terminology
* **Document:** An individual sample or unit of text (e.g., a single tweet, review, or email).
* **Corpus:** The complete collection of all documents under consideration.
* **Token:** A discrete unit of text resulting from segmentation (typically words or subwords).
* **Vocabulary:** The set of unique tokens found across the entire corpus.
* **Label / Target:** The categorical class assigned to a document (e.g., `spam` or `ham`).

### Benchmark Datasets
* **SMS Spam Collection:** 5,574 text messages labeled for binary spam detection.
* **IMDB Movie Reviews:** 50,000 highly polarized movie reviews for sentiment classification.
* **20 Newsgroups:** ~18,000 forum posts categorized across 20 distinct topic clusters.
* **AG News:** ~127,000 news articles categorized into 4 core classes.

### Inspecting Datasets with Python

```python
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Load a local CSV dataset
df = pd.read_csv("spam.csv")
print("Shape:", df.shape)
print("Class Distribution:\n", df["label"].value_counts())
print("Average Length:", df["text"].str.len().mean())

# Load Scikit-Learn built-in dataset
news = fetch_20newsgroups(subset="train")
print("Number of documents:", len(news.data))

```

> ⚠️ **Class Imbalance Warning:** If a dataset has 90% "ham" and 10% "spam", a naive model predicting only "ham" achieves 90% accuracy while failing its objective. Always verify class distributions.

---

## 5. Text Cleaning and Normalization

Raw text scraped from digital platforms contains noise that degrades model performance. **Normalization** ensures uniform representations across documents.

### Common Cleaning Operations

| Step | Raw Text | Cleaned Output |
| --- | --- | --- |
| **Lowercasing** | `"Apple IS Great"` | `"apple is great"` |
| **Remove Punctuation** | `"Wow!!! Amazing..."` | `"Wow Amazing"` |
| **Remove Digits** | `"Order 123 shipped"` | `"Order shipped"` |
| **Remove URLs** | `"Visit https://abc.com now"` | `"Visit now"` |
| **Remove HTML Tags** | `"<p>Hello World</p>"` | `"Hello World"` |
| **Remove Extra Whitespace** | `"too    many   spaces"` | `"too many spaces"` |

### Universal Cleaning Function

```python
import re
import string

def clean_text(text: str) -> str:
    text = text.lower()                                               # Convert to lowercase
    text = re.sub(r"<.*?>", "", text)                                 # Strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", "", text)                      # Remove hyperlinks
    text = re.sub(r"\d+", "", text)                                   # Remove numeric digits
    text = text.translate(str.maketrans("", "", string.punctuation)) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                          # Collapse extra spaces
    return text

# Usage:
# df["clean_text"] = df["text"].apply(clean_text)

```

---

## 6. Tokenization

**Tokenization** segments a continuous sequence of characters into discrete syntactic tokens.

### Levels of Tokenization

* **Sentence Tokenization:** Splitting paragraphs into individual sentences.
* **Word Tokenization:** Segmenting sentences into separate word units and punctuation marks.
* **Character Tokenization:** Splitting words into constituent letters/characters.
* **Subword Tokenization:** Breaking words into frequent morphemes (e.g., `["un", "happi", "ness"]`), used in BERT and GPT models to handle Out-Of-Vocabulary (OOV) words.

### Implementation

```python
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello! I'm learning NLP. It's fun, isn't it?"

# Native split (preserves trailing punctuation)
print(text.split())

# NLTK Word Tokenizer (isolates punctuation and contractions)
print(word_tokenize(text))
# Output: ['Hello', '!', 'I', "'m", 'learning', 'NLP', '.', 'It', "'s", 'fun', ',', 'is', "n't", 'it', '?']

# NLTK Sentence Tokenizer
print(sent_tokenize(text))
# Output: ['Hello!', "I'm learning NLP.", "It's fun, isn't it?"]

```

---

## 7. Stop Word Removal

**Stop words** are high-frequency structural words (*"the"*, *"is"*, *"in"*, *"at"*, *"and"*) that carry minimal semantic value for basic classification tasks.

### Benefits

* **Noise Reduction:** Filters out non-informative linguistic boilerplate.
* **Dimensionality Reduction:** Compresses vocabulary size, speeding up vectorization and model training.

### When to Retain Stop Words

* **Sentiment Analysis:** Removing *"not"* alters sentiment:
* *"This movie is **not** good"* ──► *"movie good"* (Sentiment inverted!)


* **Machine Translation / Conversational Systems:** Structural words are necessary to preserve grammatical fidelity.

### Implementation

```python
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

text = "The food was not good and the service was very slow."
tokens = word_tokenize(text.lower())

# Retain negation tokens
negation_words = {"not", "no", "nor", "n't"}
filtered_tokens = [w for w in tokens if w not in stop_words or w in negation_words]

print(filtered_tokens)
# Output: ['food', 'not', 'good', 'service', 'slow', '.']

```

---

## 8. Stemming and Lemmatization

Both techniques normalize words to their common root to consolidate vocabulary entries.

| Dimension | Stemming | Lemmatization |
| --- | --- | --- |
| **Method** | Heuristic rule-based suffix truncation | Morphological analysis via dictionary (WordNet) |
| **Output** | May produce non-words (stems) | Always returns a valid dictionary headword (lemma) |
| **Speed** | Extremely fast and lightweight | Computationally slower |
| **Context Aware** | No | Yes (requires Part-of-Speech tagging) |
| **Best Used For** | Search indexing, large-scale Information Retrieval | Chatbots, text understanding, Q&A systems |

### Morphological Comparison

| Word | Porter Stemmer | WordNet Lemmatizer |
| --- | --- | --- |
| `studies` | `studi` | `study` |
| `running` | `run` | `run` *(with `pos='v'`)* |
| `better` | `better` | `good` *(with `pos='a'`)* |
| `mice` | `mice` | `mouse` |

### Implementation

```python
import nltk
nltk.download("wordnet")
from nltk.stem import PorterStemmer, WordNetLemmatizer

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Stemmed:", ps.stem("studies"))            # Output: studi
print("Lemmatized (Noun):", lemmatizer.lemmatize("studies")) # Output: study

# Explicit Part-of-Speech (POS) Lemmatization
print("Verb:", lemmatizer.lemmatize("running", pos="v"))     # Output: run
print("Adjective:", lemmatizer.lemmatize("better", pos="a")) # Output: good

```

---

## 9. Regular Expressions (Regex)

Regular expressions define string search patterns for advanced cleaning, extraction, and validation.

### Core Metacharacters

| Symbol | Definition | Example |
| --- | --- | --- |
| `.` | Any single character except newline | `c.t` matches `cat`, `cot` |
| `\d` | Any digit `[0-9]` | `\d{10}` matches a 10-digit phone number |
| `\w` | Any alphanumeric character `[a-zA-Z0-9_]` | `\w+` matches words |
| `\s` | Any whitespace character (space, tab, newline) | `\s+` matches variable spacing |
| `+` | Quantifier: 1 or more occurrences | `\d+` matches one or more numbers |
| `*` | Quantifier: 0 or more occurrences | `ab*` matches `a`, `ab`, `abb` |
| `?` | Quantifier: 0 or 1 occurrence (optional) | `colou?r` matches `color` and `colour` |
| `^` / `$` | Start of string / End of string | `^Start`, `End$` |
| `[...]` | Character class (matches any character inside) | `[aeiou]` matches any vowel |
| `[^...]` | Negated class (matches characters NOT inside) | `[^a-z]` matches non-lowercase |

### Implementation

```python
import re

text = "Reach us at contact@ai-academy.in or support@nlp.org. Call: 9876543210 #NLP #2026"

# Extraction
emails = re.findall(r"\S+@\S+", text)
hashtags = re.findall(r"#\w+", text)
numbers = re.findall(r"\d+", text)

# Substitution / Cleaning
cleaned = re.sub(r"[^a-zA-Z\s]", "", text)

print("Emails:", emails)
print("Hashtags:", hashtags)
print("Text Only:", cleaned)

```

---

## 10. Word Frequency and Vocabulary

* **Word Frequency:** The count of occurrences of each distinct token in a document or corpus.
* **Vocabulary Size ($V$):** The total count of unique tokens across the dataset.

> **Zipf's Law:** In natural language corpora, the frequency of any word is inversely proportional to its rank in the frequency table. A small set of words (mostly stop words) accounts for the vast majority of occurrences, while most words occur rarely.

### Implementation

```python
from collections import Counter
from nltk import FreqDist

corpus = "natural language processing helps computers understand language text text"
tokens = corpus.split()

# Using standard Counter
word_counts = Counter(tokens)
print("Vocabulary Size:", len(word_counts))
print("Most Common:", word_counts.most_common(2))

# Using NLTK Frequency Distribution
fdist = FreqDist(tokens)
# fdist.plot(10) # Plots top 10 frequencies

```

---

## 11. N-grams: Unigrams, Bigrams, Trigrams

An **N-gram** is a contiguous sequence of $n$ items from a given sample of text.

$$\text{Total N-grams in a sentence} = \text{Total Tokens} - n + 1$$

Given: *"I love natural language processing"*

* **Unigram ($n=1$):** `["I", "love", "natural", "language", "processing"]`
* **Bigram ($n=2$):** `["I love", "love natural", "natural language", "language processing"]`
* **Trigram ($n=3$):** `["I love natural", "love natural language", "natural language processing"]`

### Why N-grams Matter

* **Captures Local Context:** Unigrams lose phrasing; bigrams preserve terms like *"not good"* or *"New York"*.
* **The Trade-off:** Increasing $n$ expands vocabulary exponentially, leading to sparse feature matrices. A combination of unigrams and bigrams (`ngram_range=(1, 2)`) is the standard industry baseline.

---

## 12. Bag of Words (BoW)

The **Bag of Words** model represents text as an unordered collection of word counts.

```
Documents ──► Build Global Vocabulary ──► Count Occurrences per Document ──► Sparse Matrix

```

### Example

* **$D_1$:** *"I love NLP"*
* **$D_2$:** *"I love Python and NLP"*
* **$D_3$:** *"Python is fun"*

**Vocabulary:** `["and", "fun", "i", "is", "love", "nlp", "python"]` ($V = 7$)

| Document | and | fun | i | is | love | nlp | python |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **$D_1$** | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| **$D_2$** | 1 | 0 | 1 | 0 | 1 | 1 | 1 |
| **$D_3$** | 0 | 1 | 0 | 1 | 0 | 0 | 1 |

### Limitations

1. **Loss of Word Order:** *"Dog bites man"* and *"Man bites dog"* yield identical vectors.
2. **High Sparsity:** Most values are zeros, consuming memory.
3. **No Semantic Awareness:** Words like *"great"* and *"excellent"* are treated as completely independent dimensions.

---

## 13. CountVectorizer

`CountVectorizer` is Scikit-Learn’s production implementation of the Bag of Words model.

### Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "I love NLP",
    "I love Python and NLP",
    "Python is fun"
]

# Initialize with custom constraints
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=1000
)

# Fit on training corpus and transform to matrix
X = vectorizer.fit_transform(docs)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Matrix Array:\n", X.toarray())

```

### Essential Parameters

* `stop_words='english'`: Automatically eliminates standard English stop words.
* `ngram_range=(min_n, max_n)`: Extracts unigram, bigram, or higher-order combinations.
* `max_features=K`: Retains only the top $K$ most frequent tokens.
* `min_df=int/float`: Ignores words that appear in fewer than the specified threshold of documents.
* `max_df=float`: Ignores corpus-specific common words that appear in more than $X\%$ of documents.
* `binary=True`: Records binary indicators (0 or 1) rather than frequency counts.

---

## 14. TF-IDF

**TF-IDF** (Term Frequency – Inverse Document Frequency) penalizes words that appear everywhere across a corpus while rewarding words that uniquely characterize a specific document.

### Mathematical Components

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

1. **Term Frequency (TF):** The frequency of term $t$ in document $d$.
2. **Inverse Document Frequency (IDF):** Measures term specificity across the corpus:

$$\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$

*(where $N$ is total documents, and $\text{DF}(t)$ is document frequency of term $t$)*.

### Scikit-Learn Smooth Formula

Scikit-Learn uses a smoothed variation to prevent division by zero, followed by L2 row normalization:

$$\text{IDF}_{\text{sklearn}}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$

### Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "the cat sat",
    "the dog sat",
    "the cat ran"
]

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)

print("Feature Names:", tfidf.get_feature_names_out())
print("TF-IDF Matrix:\n", X_tfidf.toarray().round(2))

```

---

## 15. CountVectorizer vs. TfidfVectorizer

| Feature | CountVectorizer | TfidfVectorizer |
| --- | --- | --- |
| **Cell Value** | Raw integer occurrences ($0, 1, 2 \dots$) | Scaled float weight ($0.0 \dots 1.0$) |
| **Common Words** | Assigned high numerical magnitude | Scaled down via IDF penalties |
| **Normalization** | None by default | L2-normalized vectors (unit length) by default |
| **Document Length Bias** | Long documents have larger counts | Invariant to length differences due to L2 scaling |
| **Optimal Classifiers** | Multinomial Naive Bayes | Logistic Regression, Linear SVM, Cosine Similarity |

### Equivalence Relation

`TfidfVectorizer` is equivalent to applying `CountVectorizer` followed by `TfidfTransformer`:

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

counts = CountVectorizer().fit_transform(docs)
tfidf_matrix = TfidfTransformer().fit_transform(counts)

```

---

## 16. Train-Test Split for Text Data

Evaluating a model on data it was trained on produces overly optimistic performance estimates due to memorization. We isolate a hold-out test set to evaluate real-world generalization.

### Preventing Data Leakage

* ❌ **Incorrect:** Applying `fit_transform` across the entire dataset before splitting. The vocabulary from the test set leaks into training statistics.
* ✅ **Correct:** Split the raw text data first. Run `fit_transform` **only** on the training subset, and run `transform` on the test subset.

### End-to-End Pipeline Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

# 1. Load Data
df = pd.read_csv("spam.csv")
X = df["text"]
y = df["label"]

# 2. Train-Test Split (with stratification to preserve class ratios)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Method A: Manual Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train) # Learn vocabulary & transform
X_test_vec = vectorizer.transform(X_test)       # Use training vocabulary only

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 4. Method B: Clean Scikit-Learn Pipeline (Best Practice)
pipeline = make_pipeline(
    TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
    MultinomialNB()
)

pipeline.fit(X_train, y_train)
print("Pipeline Score:", pipeline.score(X_test, y_test))

# Inference on unseen text
sample = ["Congratulations! You have won a $1,000 gift card. Call now!"]
print("Prediction:", pipeline.predict(sample))

```

---

## 📚 Summary Checklist

```
[✓] Text Cleaning: Lowercase, remove HTML, strip URLs & punctuation
[✓] Preprocessing: Tokenize words, handle stop words, apply lemmatization
[✓] Feature Engineering: Extract Unigrams + Bigrams via TF-IDF Vectorizer
[✓] Model Validation: Split data BEFORE fitting vectorizers to prevent data leakage
[✓] Evaluation: Track precision, recall, and F1-score alongside baseline accuracy

```

```

```
