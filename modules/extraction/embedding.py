from sentence_transformers import SentenceTransformer

# For more information on sentence_transformer package, refer to https://sbert.net/
# For list of models refer to: https://sbert.net/docs/sentence_transformer/pretrained_models.html

# THIS IS ALREADY IMPLEMENTED!

class Embedding:
    """
    A class used to encode sentences into embeddings using a specified pre-trained model.

    Attributes
    ----------
    model : SentenceTransformer
        A pre-trained sentence transformer model used for encoding.

    Methods
    -------
    encode(sentence: str) -> np.ndarray
        Encodes the given sentence into an embedding.
    """

    def __init__(self, model_name: str='all-MiniLM-L6-v2'):
        """
        Initializes the Embedding class with a specified model name.

        Parameters
        ----------
        model_name : str
            The name of the pre-trained model to be used for encoding.
            You can try different sentence encoders here: 
            - https://sbert.net/docs/sentence_transformer/pretrained_models.html
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, sentence):
        """
        Encodes the given sentence into an embedding.

        Parameters
        ----------
        sentence : str
            The sentence to be encoded.

        Returns
        -------
        np.ndarray
            The embedding of the given sentence.
        """
        return self.model.encode(sentence)
    
    def decode(self, sentence):
        """
        Decodes the given sentence into an embedding.

        Parameters
        ----------
        sentence : str
            The sentence to be decoded.

        Returns
        -------
        np.ndarray
            The string representation of the given encoded sentence.
        """
        return self.model.decode(sentence)


if __name__ == "__main__":
    # You can try different sentence encoders here: 
    # https://sbert.net/docs/sentence_transformer/pretrained_models.html
    
    embedding_model_name = 'all-MiniLM-L6-v2'
    embedding_model = Embedding(embedding_model_name)

    sentence = "Who suggested Lincoln grow a beard?"
    sentence_embedding = embedding_model.encode(sentence)

    print(sentence_embedding)
    print(sentence_embedding.shape)
