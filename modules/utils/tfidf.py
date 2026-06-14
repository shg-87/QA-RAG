import math
import re
from collections import Counter
from typing import List

import numpy as np

import nltk
from nltk.corpus import stopwords

try:
    from textwave.modules.utils.text_processing import process_text
except ImportError:
    from .text_processing import process_text


nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


class TF_IDF:
    """
    A TF-IDF transformer that learns a vocabulary from a corpus and transforms
    documents into their TF-IDF representation.
    """

    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = np.array([], dtype=float)

    def _tokenize(self, text: str):
        """
        Tokenizes the input text by converting it to lowercase, normalizing tokens,
        and removing stop words.
        """
        text = re.sub(r"(?i)\b(\w+)'s\b", r"\1", text)
        processed_text = process_text(text, use_lemmatization=True)
        tokens = re.findall(r"\b\w+\b", processed_text.lower(), flags=re.UNICODE)
        return [token for token in tokens if token not in STOP_WORDS]

    def fit(self, documents: List[str]):
        """
        Learns the vocabulary and computes the inverse document frequency (IDF).
        """
        document_frequencies = Counter()
        vocabulary = set()

        for document in documents:
            tokens = self._tokenize(document)
            vocabulary.update(tokens)
            document_frequencies.update(set(tokens))

        self.vocabulary_ = {token: index for index, token in enumerate(sorted(vocabulary))}
        self.idf_ = np.zeros(len(self.vocabulary_), dtype=float)

        total_documents = len(documents)
        for token, index in self.vocabulary_.items():
            self.idf_[index] = math.log(total_documents / (document_frequencies[token] + 1)) + 1

        return self

    def transform(self, document: str):
        """
        Transforms a document into its TF-IDF representation.
        """
        if not self.vocabulary_:
            raise AttributeError("TF_IDF must be fitted before calling transform().")

        vector = np.zeros(len(self.vocabulary_), dtype=float)
        tokens = [token for token in self._tokenize(document) if token in self.vocabulary_]

        if not tokens:
            return vector

        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        for token, count in token_counts.items():
            index = self.vocabulary_[token]
            term_frequency = count / total_tokens
            vector[index] = term_frequency * self.idf_[index]

        return vector



if __name__ == "__main__":
    # Example corpus of 9 documents to fit the TF-IDF transformer.
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A quick movement of the enemy will jeopardize six gunboats.",
        "All that glitters is not gold.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you; ask what you can do for your country.",
        "That's one small step for man, one giant leap for mankind.",
    ]

    # Fit the transformer on the corpus.
    transformer = TF_IDF()
    transformer.fit(corpus)
    
    # Test document to transform after fitting the corpus.
    test_document = "The quick dog jumps high over the lazy fox."
    tfidf_test = transformer.transform(test_document)
    
    print(tfidf_test)
