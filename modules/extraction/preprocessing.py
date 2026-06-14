from typing import List

import nltk




class DocumentProcessing:
    """
    A class used for processing documents including reading, trimming whitespace,
    and splitting documents into sentence chunks.
    """

    def __init__(self):
        """Initialize the DocumentProcessing instance (no specific setup needed)"""
        pass

    def __read_text_file(self, file_path: str) -> str:
        """
        Private helper method to read the content of a text file

        Args:
            file_path (str): Path to the text file

        Returns:
            str: Content of the file as a string, or an error message if reading fails
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return f"The file at {file_path} was not found."
        except Exception as e:
            return f"An error occurred: {e}"

    def trim_white_space(self, text: str) -> str:
        """
        Remove extra whitespace from a string by splitting and rejoining with single spaces.

        Args:
            text (str): Input text.

        Returns:
            str: Text with normalized whitespace.
        """
        return " ".join(text.split())

    def sentence_chunking(self, document_filename: str, num_sentences: int, overlap_size: int = 0) -> list:
        """
        Split a document into chunks, each containing a fixed number of sentences

        Args:
            document_filename (str): path to the document file
            num_sentences (int): number of sentences per chunk
            overlap_size (int, optional): number of overlapping sentences between consecutive chunks.
                                         Defaults to 0 (no overlap)

        Returns:
            list: List of string chunks, each containing 'num_sentences' sentences
                  If the file cannot be read, returns a list with the error message
        """

        # Read the raw text from the file
        text = self.__read_text_file(document_filename)

        # Only proceed if text was read successfully
        if isinstance(text, str):
            # Normalize whitespace
            text = self.trim_white_space(text)
            # Split into individual sentences using NLTK's sentence tokenizer
            sentences = nltk.sent_tokenize(text)

            chunks = []
            i = 0
            # Slide over the sentences with the given step
            while i < len(sentences):
                # Join the selected sentences into a single chunk
                chunk = " ".join(sentences[i:i + num_sentences])
                chunks.append(chunk)
                # Move the start index, accounting for overlap
                i += (num_sentences - overlap_size)

            return chunks
        # If reading failed, return the error message wrapped in a list
        return [text]

    def fixed_length_chunking(self, document_filename: str, chunk_size: int, overlap_size: int = 2) -> List[str]:
        """
        Divides the document into fixed-size chunks of characters, with optional overlap

        Args:
            document_filename (str): Path to the document file.
            chunk_size (int): Number of characters per chunk. Must be > 0.
            overlap_size (int, optional): Number of characters to overlap between chunks.
                                         Must be >= 0 and < chunk_size. Defaults to 2.

        Returns:
            List[str]: List of character-based chunks.

        Raises:
            ValueError: If chunk_size <= 0, overlap_size < 0, or overlap_size >= chunk_size.
        """
        # Validate input parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if overlap_size < 0:
            raise ValueError("overlap_size must be non-negative.")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size.")

        # Read and clean the document text
        text = self.__read_text_file(document_filename)
        text = self.trim_white_space(text)

        # Return empty list if the document is empty after trimming
        if text == "":
            return []

        chunks = []
        step = chunk_size - overlap_size   # Number of characters to advance each iteration

        # Iterate over the text, creating overlapping chunks
        for start in range(0, len(text), step):
            chunk = text[start:start + chunk_size]
            if chunk:   # Only add non-empty chunks
                chunks.append(chunk)
            # Stop when the next chunk would start beyond the end of the text
            if start + chunk_size >= len(text):
                break

        return chunks


if __name__ == "__main__":
    processing = DocumentProcessing()

    # Example to split documents into sentence chunks
    chunks = processing.sentence_chunking("storage/corpus/S08_set3_a1.txt.clean", num_sentences=5, overlap_size=3)
    for idx, chunk in enumerate(chunks):
        print(idx, chunk)
