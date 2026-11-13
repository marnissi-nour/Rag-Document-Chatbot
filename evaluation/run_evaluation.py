import json
from pathlib import Path
from typing import Any

from app.rag import RAGService


def load_questions() -> list[dict[str, Any]]:
    path = Path(__file__).with_name("questions.json")
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> None:
    questions = load_questions()
    rag = RAGService()
    results: list[dict[str, Any]] = []

    for item in questions:
        expected_sources = {
            source.casefold() for source in item["expected_sources"]
        }
        expected_terms = {
            term.casefold() for term in item["expected_terms"]
        }
        retrieved = rag._search(item["question"], top_k=5)
        retrieved_sources = {
            chunk["source"].casefold() for chunk, _score in retrieved
        }
        answer = rag.answer(item["question"], top_k=5)["answer"].casefold()
        source_hit = bool(expected_sources & retrieved_sources)
        covered_terms = {
            term for term in expected_terms if term in answer
        }
        results.append(
            {
                "id": item["id"],
                "source_recall": source_hit,
                "term_coverage": (
                    len(covered_terms) / len(expected_terms)
                    if expected_terms
                    else 1.0
                ),
                "retrieved_sources": sorted(retrieved_sources),
                "answer": answer,
            }
        )

    source_recall = sum(result["source_recall"] for result in results) / len(results)
    term_coverage = sum(result["term_coverage"] for result in results) / len(results)
    print(
        json.dumps(
            {
                "questions": len(results),
                "source_recall_at_5": round(source_recall, 3),
                "answer_term_coverage": round(term_coverage, 3),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run()
