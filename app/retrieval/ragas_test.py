"""
Minimal RAGAS evaluation scaffold.

This file evaluates retrieval using RAGAS only if you install RAGAS.
It does NOT affect normal retrieval.py execution.

For serious evaluation, replace the example reference answers/contexts
with the exact ground-truth answers and source chunks from your KB.
"""

from app.retrieval.retrieval import UnstructuredRetriever, format_context

from app.logging_config import get_logger, trace

logger = get_logger(__name__)

TEST_CASES = [
    {
        "question": "How does auto-debit work for insurance premiums?",
        "reference_answer": "Use the approved insurance policy and FAQ documents as the source of truth.",
    },
    {
        "question": "What does policy POL-GE-30250 say about auto-debit?",
        "reference_answer": "Use the exact policy document identified by POL-GE-30250 as the source of truth.",
    },
]

@trace(logger)
def main():
    retriever = UnstructuredRetriever.from_chroma()

    for case in TEST_CASES:
        result = retriever.retrieve(case["question"])

        print("\n" + "=" * 80)
        print("QUESTION:", case["question"])
        print("STATUS:", result.retrieval_status)
        print("CONFIDENCE:", result.confidence)
        print("REFERENCE ANSWER:", case["reference_answer"])
        print("\nRETRIEVED CONTEXT:")
        print(format_context(result))

if __name__ == "__main__":
    main()
