from rag.document_loader import load_documents
from models.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from models.llm import LLM
from evaluation.hallucination_detector import HallucinationDetector


KNOWLEDGE_FILE = "data/knowledge_base.txt"


def build_rag():

    print("\nLoading documents...")

    documents = load_documents(
        KNOWLEDGE_FILE
    )

    print(f"Documents loaded: {len(documents)}")

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(
        documents
    )

    dimension = embeddings.shape[1]

    vector_store = VectorStore(
        dimension
    )

    vector_store.add_documents(
        documents,
        embeddings
    )

    retriever = Retriever(
        embedding_model,
        vector_store
    )

    llm = LLM()

    detector = HallucinationDetector()

    return retriever, llm, detector


def ask_question(
    question,
    retriever,
    llm,
    detector
):

    retrieved = retriever.retrieve(
        question
    )

    context = "\n\n".join(
        item["document"]
        for item in retrieved
    )

    answer = llm.generate(
        question,
        context
    )

    evaluation = detector.evaluate(
        answer,
        context
    )

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "score": evaluation["score"],
        "label": evaluation["label"]
    }


def main():

    retriever, llm, detector = build_rag()

    print("\n================================")
    print("LLM HALLUCINATION DETECTION")
    print("================================")

    while True:

        question = input(
            "\nAsk a question (type exit to stop): "
        )

        if question.lower() == "exit":
            break

        result = ask_question(
            question,
            retriever,
            llm,
            detector
        )

        print("\nQuestion:")
        print(result["question"])

        print("\nRetrieved Context:")
        print(result["context"])

        print("\nLLM Answer:")
        print(result["answer"])

        print("\nHallucination Analysis:")
        print(
            f"Consistency Score: "
            f"{result['score']}"
        )

        print(
            f"Result: {result['label']}"
        )


if __name__ == "__main__":
    main()