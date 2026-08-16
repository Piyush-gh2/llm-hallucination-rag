import re


def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    documents = re.split(r"\n\s*\n", text)

    documents = [
        document.strip()
        for document in documents
        if document.strip()
    ]

    return documents