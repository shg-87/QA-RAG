import re

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

# THIS IS ALREADY IMPLEMENTED!

def process_text(text, use_stemming=False, use_lemmatization=False):
    """
    Process and normalize input text by tokenizing and optionally applying stemming and lemmatization.
    
    This function converts the input text to lowercase and uses a regular expression to extract 
    alphanumeric word tokens. Optionally, if stemming is enabled, the function applies the PorterStemmer 
    to reduce words to their root form. If lemmatization is enabled, the function applies the 
    WordNetLemmatizer to convert words into their canonical form. When both stemming and lemmatization 
    are enabled, stemming is applied first, followed by lemmatization.
    
    Parameters:
        text (str): The input text to be processed. It should be a string containing the text data.
        use_stemming (bool): Flag indicating whether to apply stemming. Default is False.
        use_lemmatization (bool): Flag indicating whether to apply lemmatization. Default is False.
    
    Returns:
        str: A single string containing the processed text, with tokens joined by a space.
    
    Example:
        >>> sample_text = "Running runners run quickly."
        >>> processed = process_text(sample_text, use_stemming=True)
        >>> print(processed)
        run runner run quickli
    
    Notes:
        - NLTK's resources (e.g., "wordnet") is required to use lemmatization.
        - The order of operations is: lowercase conversion, tokenization, optional stemming, then optional lemmatization.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

if __name__ == "__main__":
    # Sample input text for demonstration
    sample_text = "Running runners run quickly towards the finish line."
    
    # Process text without stemming or lemmatization
    processed_plain = process_text(sample_text)
    print("Processed without stemming/lemmatization:")
    print(processed_plain)
    
    # Process text with stemming enabled
    processed_stem = process_text(sample_text, use_stemming=True)
    print("\nProcessed with stemming:")
    print(processed_stem)
    
    # Process text with lemmatization enabled
    processed_lemma = process_text(sample_text, use_lemmatization=True)
    print("\nProcessed with lemmatization:")
    print(processed_lemma)
    
    # Process text with both stemming and lemmatization enabled
    processed_both = process_text(sample_text, use_stemming=True, use_lemmatization=True)
    print("\nProcessed with both stemming and lemmatization:")
    print(processed_both)
