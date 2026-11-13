# RAG evaluation

`questions.json` is a small ground-truth dataset. Replace its sample rows with
questions about an indexed document before measuring the application.

Each row contains:

- `question`: the user question
- `expected_sources`: filenames that should be retrieved
- `expected_terms`: important terms that should appear in the answer

Run the evaluation from the project root:

```powershell
python evaluation\run_evaluation.py
```

The script reports:

- **Source recall@k**: percentage of questions where at least one expected
  source appears in the top-k retrieved chunks
- **Answer term coverage**: percentage of expected terms found in the generated
  answer
- Per-question results for investigating failures

This is a lightweight regression check, not a replacement for human review.
Update the dataset whenever the indexed document set changes.
