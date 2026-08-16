from transformers import pipeline

from config import LLM_MODEL


class LLM:

    def __init__(self):

        print("Loading language model...")

        self.generator = pipeline(
            "text2text-generation",
            model=LLM_MODEL
        )

    def generate(self, question, context):

        prompt = f"""
Answer the question using ONLY the information provided
in the context.

If the answer is not present in the context, say:
"I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""

        result = self.generator(
            prompt,
            max_new_tokens=100,
            do_sample=False
        )

        return result[0]["generated_text"]