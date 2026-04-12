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
