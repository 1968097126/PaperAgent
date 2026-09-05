# Evaluation data

The five JSONL files in this directory are **synthetic evaluation annotations**
generated deterministically from the repository's real corpus chunks. They are not
human-labelled data and must not be described as such. Every retrieval target is a
real `chunk_id` from `corpus/chunks.jsonl`; the strict validator rejects missing IDs,
distribution drift, malformed records, and E rows that differ from their source M turn.

Regenerate and validate with seed 42:

```bash
conda run -n nanocc python -m eval.run_evaluation \
  --suite datasets --build-datasets --seed 42
conda run -n nanocc python -m eval.validate_datasets \
  --data-dir eval/data --chunks corpus/chunks.jsonl
```

Do not treat metrics on these synthetic annotations as human-reviewed benchmark
results. Human review is a separate Phase 7 requirement. Generate the protected review
package with `python -m eval.review_annotations export` and follow
[`eval/review/README.md`](../review/README.md); exporting the package does not change
this provenance declaration.
