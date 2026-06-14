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



# Keep preprocessing lightweight and deterministic.
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


class BagOfWords:
    """
    A Bag-of-Words representation transformer that learns a vocabulary from a corpus
    and transforms documents into their Bag-of-Words (BoW) representation.
    """

    def __init__(self):
        self.vocabulary_ = {}

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
        Learns the vocabulary from the corpus.
        """
        vocabulary = set()
        for document in documents:
            vocabulary.update(self._tokenize(document))

        self.vocabulary_ = {token: index for index, token in enumerate(sorted(vocabulary))}
        return self

    def transform(self, document: str):
        """
        Transforms a single document into its Bag-of-Words representation.
        """
        if not self.vocabulary_:
            raise AttributeError("BagOfWords must be fitted before calling transform().")

        vector = np.zeros(len(self.vocabulary_), dtype=float)
        counts = Counter(token for token in self._tokenize(document) if token in self.vocabulary_)

        for token, count in counts.items():
            vector[self.vocabulary_[token]] = float(count)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector



if __name__ == "__main__":
    # Example corpus of 9 documents to train the Bag-of-Words representation.
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

    # Fit the transform on the corpus.
    transform = BagOfWords()
    transform.fit(corpus)
    
    # Test document to transform after fitting the corpus.
    test_document = "The quick dog jumps high over the lazy fox."
    bow_test = transform.transform(test_document)
    
    print(bow_test)
