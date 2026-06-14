import os
import time
import numpy as np
from typing import List, Tuple, Union

from mistralai import Mistral

# THIS IS ALREADY IMPLEMENTED!

class QAGeneratorMistral:
    """
    A question-answer generator that uses the Mistral API to generate answers
    based on provided context and a query.
    """

    def __init__(self, api_key, temperature=0.3, generator_model="mistral-small-latest"):
        """
        Initializes the QA_Generator class with API key, temperature, and model.

        :param api_key: A string containing the API key for Mistral API authentication.
        :param temperature: A float specifying the randomness of the answer generation.
        :param generator_model: A string specifying the generator model name to use.
        """
        self.api_key = api_key
        self.temperature = temperature
        self.generator_model = generator_model
        self.client = Mistral(api_key=api_key)

    def generate_answer(self, query, context):
        """
        Generates an answer based on the provided query and context.

        :param query: A string containing the question to be answered.
        :param context: A list of strings representing the context in which
                        the question should be answered.
        :return: A string containing the generated answer.
        """
        combined_input = (
            f"Question: {query}\n\n"
            f"Context: {', '.join(context)}\n\n"
        )
        chat_response = self.client.chat.complete(
            model=self.generator_model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You must answer the user's questions only based "
                        "on the provided context. Do not use any external or prior knowledge. "
                        "Provide clear, concise, and full-sentence answers."
                        "If the context does not mention the answer, respond with 'No context'."
                    )
                },
                {
                    "role": "user",
                    "content": combined_input,
                },
            ]
        )

        # Introduce a delay for throttling or rate-limiting purposes. 
        time.sleep(2)

        # print(chat_response.choices)

        return chat_response.choices[0].message.content
    

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Set up Mistral-based generator and ask the question
    # ------------------------------------------------------------------

    # Read your Mistral API key from the environment.
    # (Make sure `export MISTRAL_API_KEY=...` is set in your shell.)
    API_KEY = os.environ["MISTRAL_API_KEY"]

    # Instantiate the wrapper we implemented earlier for the Mistral API.
    generator_mistral = QAGeneratorMistral(api_key=API_KEY)

    # Context passages that the model must rely on.
    context = [
        "Albert Einstein was a theoretical physicist born in Germany.",
        "He developed the theory of relativity, one of the two pillars of modern physics.",
        "Einstein was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect."
    ]

    # The question we want answered *solely* from the context above.
    query = "What was Einstein awarded the Nobel Prize for?"

    # Ask the question via the Mistral LLM (will query the Mistral service).
    answer = generator_mistral.generate_answer(query=query, context=context)

    # Pretty-print the result.
    print("=== Mistral generator ===")
    print(f"Question: {query}")
    print(f"Answer:   {answer}\n")
