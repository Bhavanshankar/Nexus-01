from app.retrieval.retrieval import UnstructuredRetriever, format_context

from app.logging_config import get_logger, trace

logger = get_logger(__name__)

@trace(logger)
def main():
    retriever = UnstructuredRetriever.from_chroma()

    print("Nexus Unstructured Retrieval Demo")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        result = retriever.retrieve(query)

        print("\nStatus:", result.retrieval_status)
        print("Confidence:", round(result.confidence, 4))
        print("Safe to answer:", result.safe_to_answer)
        print("Reason:", result.reason)

        if result.retrieval_status == "INSUFFICIENT_CONTEXT":
            print("\nSAFE ABSTENTION:")
            print("The knowledge base does not contain enough verified context.")
        else:
            print("\nRetrieved context:")
            print(format_context(result))

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
