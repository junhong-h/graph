## Approach

- Think before acting. Read relevant existing files before writing code.
- Keep solutions simple, modular, and easy to change.
- Prefer editing over rewriting whole files.
- Test code before declaring done.
- User instructions override this file.
- Always respond in Chinese unless the user explicitly requests another language.

## Research Code Rules

- Separate config from code.
- Avoid tight coupling between data, model calls, prompts, pipeline logic, and evaluation.
- Save outputs, metrics, and key logs for each run.
- Save intermediate results when useful, not just final scores.
- Make prompt, model, and evaluation changes explicit.
- Assume long runs can fail; support checkpointing and resume when possible.
- Handle sample-level failures without crashing the whole run.
- Do not silently ignore errors.
- Do not hide core logic in notebooks.
- Do not turn `utils.py` into a dumping ground.

## Experiment Rules

Every experiment must live in `experiments/YYYY-MM-DD-NNN-<slug>/`. See `experiments/README.md` for the naming convention and the full experiment index.

### Before running

1. Copy the template: `cp -r experiments/.template experiments/YYYY-MM-DD-NNN-<slug>`
2. Fill in `config.yaml`: set `experiment.id/name/description/branch/commit/based_on` and `sample_ids`.
3. Fill in `notes.md`: write the hypothesis and list the code changes before running — not after.
4. Commit `config.yaml` and `notes.md` before starting the run.

### Running

```bash
python scripts/build_memory.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
python scripts/run_qa.py       --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
```

- Never reuse an existing experiment directory for a different run. If you tweak config and re-run, create a new `NNN+1` directory.
- The old `--config` interface still works for quick one-off tests, but results must not be the basis for any conclusion unless saved into an experiment directory.

### After running

1. Run summarize script to generate `build_stats.json` and `qa_analysis.xlsx`:
   ```bash
   python scripts/summarize_exp.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
   ```
2. Fill in the Graph Stats and QA Results tables in `notes.md` (from `build_stats.json` and `qa_metrics.json`).
3. Write the Analysis section: did results match the hypothesis? Why / why not?
4. Update `experiments/README.md` experiment index table.
5. Commit: `git add experiments/YYYY-MM-DD-NNN-<slug> && git commit`.

### What gets tracked in git

| Path | Tracked | Reason |
|------|---------|--------|
| `experiments/*/config.yaml` | ✓ | reproducibility |
| `experiments/*/notes.md` | ✓ | analysis record |
| `experiments/*/build/graphs/*.json` | ✓ | ~200 KB each, needed for ablations |
| `experiments/*/build/build_stats.json` | ✓ | graph + trajectory stats summary |
| `experiments/*/qa/qa_metrics.json` | ✓ | accuracy/F1 summary |
| `experiments/*/qa/qa_analysis.xlsx` | ✓ | per-question QA review table |
| `experiments/*/chroma/` | ✗ | 67 MB+, rebuildable from graphs |
| `experiments/*/build/*.log` | ✗ | large, not essential |
| `experiments/*/qa/qa_results*.jsonl` | ✗ | large, rebuildable |

### Baseline comparison

- Always state which experiment ID is the baseline in `config.yaml` (`experiment.based_on`).
- When comparing, note whether the graph is the same or different between runs — a graph change confounds algorithm changes.
- Single-sample results (e.g. conv-26 only) are exploratory. Full-10-sample runs are needed for conclusions.
- The canonical baseline is documented in `docs/report_progress_apr20.md` (P0+P1, Qwen3-4B, Cat1-4 Acc=87.3%).

## Report Rules

- Experiment-level notes live in `experiments/.../notes.md` (filled per experiment).
- Cross-experiment analysis and milestone reports live in `docs/report_TOPIC_MMMYY.md`.
- Do not create a new `docs/` report for a single experiment — use `notes.md`.
- Every `docs/` report must include a comparison table with at least one prior result.
