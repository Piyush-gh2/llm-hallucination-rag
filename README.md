# LLM Hallucination Detection Using RAG

## Abstract

This project investigates whether Retrieval-Augmented Generation
can reduce hallucinations in Large Language Models.

## Research Question

Does RAG improve factual consistency in LLM-generated answers?

## Hypothesis

Providing an LLM with relevant information retrieved from a trusted
knowledge base will reduce unsupported responses.

## Methodology

1. Create a trusted knowledge base.
2. Convert documents into embeddings.
3. Store embeddings using FAISS.
4. Retrieve relevant documents.
5. Provide retrieved documents to the LLM.
6. Generate an answer.
7. Evaluate the answer against the retrieved evidence.
8. Calculate hallucination metrics.

## Technologies

- Python
- Transformers
- Sentence Transformers
- FAISS
- PyTorch
- Scikit-learn
- NLP

## Evaluation

The experiment measures:

- Hallucination rate
- Factual consistency
- Retrieval relevance
- Supported responses
- Unsupported responses

## Conclusion

The experiment evaluates whether grounding LLM responses with
retrieved evidence can improve factual reliability.
