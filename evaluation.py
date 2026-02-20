import json
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from vector_store import get_embedding_model, get_vector_store, search_similar_chunks


@dataclass
class RetrievalTestCase:
    """Represents a single test case for retrieval evaluation."""

    query: str
    expected_page: int


def load_test_cases_from_json(file_path: str) -> list[RetrievalTestCase]:
    """Loads the evaluation dataset from an external JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data: list[dict] = json.load(file)

    test_cases: list[RetrievalTestCase] = []
    for item in raw_data:
        test_case = RetrievalTestCase(
            query=item["query"], expected_page=item["expected_page"]
        )
        test_cases.append(test_case)

    return test_cases


def evaluate_retrieval(
    vector_store: Chroma, test_cases: list[RetrievalTestCase], k: int = 3
) -> float:
    """Runs evaluation against a list of test cases and returns the Hit Rate."""
    successful_hits: int = 0
    total_tests: int = len(test_cases)

    for index, test in enumerate(test_cases, start=1):
        print(f"\n[Test {index}/{total_tests}] Query: '{test.query}'")

        results: list[tuple[Document, float]] = search_similar_chunks(
            vector_store=vector_store, query=test.query, k=k
        )

        retrieved_pages: list[int] = []
        for doc, _ in results:
            page_idx: int = int(doc.metadata.get("page", -1))
            if page_idx != -1:
                page_num: int = page_idx + 1
            else:
                page_num = -1
            retrieved_pages.append(page_num)

        is_hit: bool = test.expected_page in retrieved_pages

        if is_hit:
            successful_hits += 1
            print(
                f"✅ PASS! Expected page {test.expected_page} found. (Pages retrieved: {retrieved_pages})"
            )
        else:
            print(
                f"❌ FAIL! Expected page {test.expected_page} NOT found. (Pages retrieved: {retrieved_pages})"
            )

    hit_rate: float = (successful_hits / total_tests) * 100
    print(f"\n{'=' * 50}")
    print(f"EVALUATION COMPLETE. HIT RATE: {hit_rate:.1f}%")
    print(f"{'=' * 50}")

    return hit_rate


def main() -> None:
    embedding_model: HuggingFaceEmbeddings = get_embedding_model(
        "sdadas/mmlw-retrieval-roberta-large"
    )
    vector_db: Chroma = get_vector_store(
        embedding_model=embedding_model, collection_name="general"
    )

    dataset_path: str = "golden_dataset.json"
    test_cases: list[RetrievalTestCase] = load_test_cases_from_json(dataset_path)

    evaluate_retrieval(vector_store=vector_db, test_cases=test_cases, k=5)


if __name__ == "__main__":
    main()
