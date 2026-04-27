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

Formal experiments must live in `experiments/YYYY-MM-DD-NNN-<slug>/`.
Use `runs/` only for quick one-off checks; do not base research conclusions on
outputs that were not saved into an experiment directory.

### Before running

1. Copy the template: `cp -r experiments/.template experiments/YYYY-MM-DD-NNN-<slug>`.
2. Fill in `config.yaml`, including `experiment.id/name/description/branch/commit/based_on`
   and `sample_ids`.
3. Fill in `notes.md` with the hypothesis and code/config changes before running.
4. Commit `config.yaml` and `notes.md` before starting a formal run.

### Running

```bash
python scripts/build_memory.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
python scripts/run_qa.py       --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
python scripts/summarize_exp.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
```

- Never reuse an existing experiment directory for a different run. If config,
  prompts, code, model, or sample selection changes, create a new `NNN+1`
  directory.
- With `--exp-dir`, build outputs go under `experiments/.../build/`, QA outputs
  go under `experiments/.../qa/`, and Chroma goes under `experiments/.../chroma/`.
- `graph_trajectories_{sample_id}.jsonl` is written directly under
  `experiments/.../build/`, not under `build/graphs/`.

### After running

1. Run `python scripts/summarize_exp.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>`.
2. Fill in graph stats and QA result tables in `notes.md`.
3. Write the analysis section, including whether results matched the hypothesis.
4. Update `experiments/README.md`.
5. Commit the experiment artifacts that are meant to be tracked.
