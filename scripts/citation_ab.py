"""A/B test: does the instruction placed right before "Answer:" make the model cite its sources?

For each eval question it retrieves the same top-k sources once, then asks the LLM twice:
  A  v2.0's original prompt: sources + question + "Answer:" (citation rules only in the system prompt)
  B  the current prompt (llm.build_rag_prompt), which adds the citation instruction before "Answer:"
Both use the same system prompt and the current citation parser. Prints one JSON line.

Runs inside the gateway container (it needs the models, Qdrant and Ollama):
    docker compose exec -T -e DAM_QUESTIONS="$(cat eval/questions.json)" api-gateway \
        python - < scripts/citation_ab.py >> benchmarks/results.jsonl
"""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime

from app import embeddings, llm, vectors
from app.main import format_hit, source_label

K = 5


def original_prompt(question: str, sources: list[dict]) -> str:
    blocks = [f"[{s['n']}] ({s['label']})\n{s['text']}" for s in sources]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


async def main() -> None:
    questions = json.loads(os.environ["DAM_QUESTIONS"])["text"]
    arms = {"original_prompt": original_prompt, "current_prompt": llm.build_rag_prompt}
    counts = {arm: {"cited_any": 0, "cited_expected_file": 0} for arm in arms}
    rows = []
    for item in questions:
        vector = embeddings.embed_texts([item["q"]])[0]
        hits = vectors.search(vectors.TEXT, vector, K, None, 0.2)
        sources = [{"n": i + 1, "label": source_label(h.payload or {}), **format_hit(h)} for i, h in enumerate(hits)]
        row = {"q": item["q"]}
        for arm, build in arms.items():
            answer = await llm.chat(llm.RAG_SYSTEM, build(item["q"], sources))
            cited = llm.cited_numbers(answer, len(sources))
            files = {s["filename"] for s in sources if s["n"] in cited}
            counts[arm]["cited_any"] += bool(cited)
            counts[arm]["cited_expected_file"] += item["file"] in files
            row[arm] = {"cited": cited, "answer": answer[:160]}
        rows.append(row)
        print(
            f"{item['q'][:55]:<55} A={row['original_prompt']['cited']} B={row['current_prompt']['cited']}",
            file=sys.stderr,
        )
    n = len(questions)
    print(
        json.dumps(
            {
                "kind": "citation_ab",
                "api": "v2",
                "at": datetime.now(UTC).isoformat(),
                "label": "v2.0.2-citation-ab",
                "k": K,
                "questions": n,
                "original_prompt": {k: f"{v}/{n}" for k, v in counts["original_prompt"].items()},
                "current_prompt": {k: f"{v}/{n}" for k, v in counts["current_prompt"].items()},
                "rows": rows,
            }
        )
    )


asyncio.run(main())
