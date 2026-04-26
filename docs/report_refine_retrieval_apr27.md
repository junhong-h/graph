# GraphMemory Refinement Report: Construction and Retrieval

**Date**: 2026-04-27  
**Branch**: `refine-retrieval-construction`  
**Evaluation run**: `runs/refine_final_conv41/`  
**Sample**: `conv-41`  
**Questions**: 193  
**Judge**: LLM judge via `scripts/run_qa.py`

This report summarizes the code changes made from the refine plan, the final
`conv-41` QA result, the observed failure modes, and the next recommended
improvements.

---

## 1. What Was Implemented

The branch contains six staged commits:

| Commit | Purpose |
|---|---|
| `651727f` | Fixed QA retrieval baseline behavior, including final answer compression and answerable-refusal repair. |
| `f435386` | Added construction invariants: normalized edge families, construction context, event metadata repair, and orphan-event linking. |
| `bc620d5` | Refined construction ontology prompt: clearer distinction between Entity, Concept, and Event. |
| `c85036e` | Added multi-seed retrieval localization for recall-heavy questions. |
| `9ec320e` | Improved jump expansion ranking, constraint scoring, frontier-exhausted fallback, and raw fallback repair. |
| `2592e5b` | Added retrieval trace diagnostics and summary script. |

The final code was pushed to:

```text
origin/refine-retrieval-construction
```

Final local test result:

```text
108 passed
```

---

## 2. Final Evaluation Result

Full QA and LLM judge were run on `conv-41`.

| Category | Total | Accuracy | Avg F1 | BLEU-1 |
|---|---:|---:|---:|---:|
| Cat1 | 31 | 93.5% | 0.4169 | 0.2929 |
| Cat2 | 27 | 88.9% | 0.3526 | 0.2585 |
| Cat3 | 8 | 62.5% | 0.1705 | 0.1667 |
| Cat4 | 86 | 90.7% | 0.5294 | 0.4667 |
| Cat5 | 41 | 53.7% | 0.4815 | 0.4749 |
| **Overall** | **193** | **81.9%** | **0.4616** | **0.3990** |

Important files:

```text
runs/refine_final_conv41/qa_results.jsonl
runs/refine_final_conv41/qa_results_eval.jsonl
runs/refine_final_conv41/qa_metrics.json
runs/refine_final_conv41/retrieval_trace_summary.json
```

---

## 3. Retrieval Trace Summary

The trace summary shows that the system is no longer simply failing because of
one isolated max-hop issue. The current bottleneck is mixed: partial graph
navigation, aggressive raw fallback, answer synthesis, and unanswerable
calibration.

Overall trace statistics:

| Metric | Value |
|---|---:|
| Avg trace steps | 2.79 |
| Samples with forced finish | 19.2% |
| Samples with frontier exhausted | 27.5% |
| Samples with raw fallback | 57.0% |
| `max_hop_exhausted` forced finishes | 19 |
| `frontier_exhausted` forced finishes | 18 |

Action counts:

| Action | Count |
|---|---:|
| `finish` | 156 |
| `raw_fallback` | 154 |
| `jump` | 124 |
| `frontier_exhausted` | 64 |
| `forced_finish` | 37 |
| `answerable_refusal_raw_fallback` | 4 |

By category, Cat5 is the most problematic trace-wise:

| Category | Samples with raw fallback | Samples with forced finish | Main signal |
|---|---:|---:|---|
| Cat1 | 14 | 5 | Aggregation still depends on raw fallback. |
| Cat2 | 17 | 4 | Time questions often need fallback despite high judge accuracy. |
| Cat3 | 6 | 5 | Small category, but forced finish ratio is high. |
| Cat4 | 46 | 9 | Many successful answers still rely on fallback. |
| Cat5 | 27 | 14 | Unanswerable questions often keep searching or hallucinate from nearby evidence. |

---

## 4. Observed Problems

### 4.1 Max-hop is still visible, but it is not the only root cause

Before this refinement, hitting max-hop looked like the main retrieval failure.
After adding frontier-exhausted handling and raw fallback, many previously stuck
paths now continue and sometimes produce correct answers. However, 19.2% of
samples still end in forced finish.

The important distinction is:

- Some forced finishes are acceptable for Cat5 when the system correctly refuses.
- Some forced finishes happen because the planner keeps choosing low-value jumps.
- Some non-forced traces still answer incorrectly because they retrieve a nearby
  but wrong event.

So the next optimization should not simply increase `max_hops`. More hops would
increase noise and make Cat5 hallucination worse.

### 4.2 Raw fallback is useful, but overused

57.0% of samples used raw fallback. This is a strong safety net and helped
several answerable questions after graph navigation failed. For example, some
date-constrained Cat4 questions reached `frontier_exhausted`, then raw fallback
recovered the right evidence.

But raw fallback also introduces false positives for Cat5. When a question is
unanswerable, raw retrieval often finds semantically similar evidence that does
not satisfy the full constraint. The final answer model then sometimes answers
from that near miss instead of refusing.

Observed pattern:

```text
Question asks about X under constraint A.
Raw fallback finds X under constraint B.
Final answer uses X anyway.
```

This is especially visible in Cat5.

### 4.3 Cat5 refusal is the weakest part

Cat5 accuracy is only 53.7%, much lower than Cat1, Cat2, and Cat4. The current
pipeline is optimized for recall-heavy answerable questions, but Cat5 requires
constraint satisfaction and negative evidence.

Examples observed during the run:

| QA | Problem |
|---|---|
| `conv-41_q154` | Asked who inspired John to start volunteering; system answered `Maria's aunt` although the target fact was not mentioned. |
| `conv-41_q167` | Asked about Maria's dinner spread with her father; system answered `banana split sundae` from a nearby food mention. |
| `conv-41_q186` | Asked about the cause of Maria's 5K charity run; system answered `veterans and their families`, which belonged to John's 5K event. |

These are not pure retrieval misses. They are failures to verify whether the
retrieved evidence satisfies all constraints in the question.

### 4.4 Similar events are being conflated

The clearest example is the repeated `5K charity run` pattern. There are multiple
events with similar surface forms but different actors, dates, and causes.

Observed errors:

- John's 5K charity run cause was answered as `homeless shelter`, which belongs
  to another similar event.
- Maria's 5K charity run was answered with `veterans and their families`, which
  belongs to John's event.

This suggests that event identity is currently too weak. A reusable event should
not be identified only by activity text. It needs a compact signature:

```text
event_signature = actor + action + object + time + location + purpose/cause
```

Retrieval should prefer events whose signature satisfies the question constraints,
not merely events whose text is semantically similar.

### 4.5 Time questions still confuse session date and event date

Some Cat2 questions are judged correct overall, but several observed failures
show that the answer model sometimes returns the conversation/session date
instead of the actual event time.

Example observed:

```text
Question: When did John participate in a 5K charity run?
Gold: first weekend of August 2023
Pred: 2023-04-07
```

This means construction and retrieval need to distinguish:

- `turn_time`: when the conversation happened.
- `event_time`: when the described event happened.
- `relative_time`: phrases like "last weekend", "the week before", "later that evening".

Currently these can collapse into a single time-like field during answer
synthesis.

### 4.6 Answer synthesis sometimes loses necessary detail

Some retrieved evidence appears relevant, but the final answer is too compressed
or too generic.

Examples observed:

| QA | Gold | Prediction | Issue |
|---|---|---|---|
| `conv-41_q148` | `doing great - learning commands and house training` | `Adjusting well` | Correct direction, missing details. |
| `conv-41_q125` | `the resilience of the veterans and their inspiring stories` | `John appreciates the veteran's hospital visit` | Restates the question, does not extract the answer. |

This is not primarily a graph issue. The final answer prompt needs stronger
instructions to preserve answer-bearing details and avoid generic paraphrases.

### 4.7 Cat3 remains weak despite multi-seed localization

Cat3 accuracy is 62.5%, with low F1 and BLEU-1. The category is small in this
sample, but its trace profile is still concerning:

```text
Cat3 samples with forced finish: 5 / 8
Cat3 samples with raw fallback: 6 / 8
```

This suggests that multi-seed localization helped but did not fully solve
open-ended aggregation/inference. Cat3 needs better evidence grouping and
answer-time synthesis, not only more nodes.

---

## 5. Interpretation

The refinement improved the retrieval behavior structurally:

- Jump expansion is now ranked instead of taking arbitrary early neighbors.
- Constraint matching is no longer a no-op.
- Frontier exhaustion now triggers raw fallback instead of silently stopping.
- Cat1/Cat3 can use a union of localized subgraphs.
- The trace diagnostics now make failure modes observable.

However, the current system still has a mismatch between two goals:

1. Answerable questions need broad recall across sessions and related events.
2. Unanswerable questions need strict constraint verification and refusal.

Raw fallback helps the first goal but hurts the second if the final answer model
does not verify evidence sufficiency.

The next improvement should therefore be less about adding more retrieval breadth
and more about adding evidence validation.

---

## 6. Recommended Next Improvements

### Priority 1: Add evidence sufficiency verification before final answer

Add a lightweight verification step before final answer generation:

```text
Given question + candidate evidence:
1. Extract required constraints from the question.
2. Check whether the evidence satisfies each constraint.
3. If any required constraint is missing or contradicted, mark evidence insufficient.
4. Only answer if sufficient; otherwise return Not mentioned in the conversation.
```

This is especially important for Cat5 and for similar-event cases.

Recommended implementation:

- Add a `verify_evidence` action or internal pre-answer verifier.
- Return structured fields:
  - `sufficient: true/false`
  - `missing_constraints`
  - `supporting_evidence_ids`
  - `reason`
- For Cat5-like or low-confidence cases, require explicit sufficient evidence
  before allowing a concrete answer.

### Priority 2: Strengthen event signatures during construction

For every event node, store normalized slots where possible:

```text
actor
action
object
time
location
purpose/cause
source_turn_ids
speaker
polarity/status
```

Then use these slots during retrieval scoring. Similar surface forms should not
merge or outrank each other unless their actor/time/purpose constraints also
match.

This directly targets the 5K charity run confusion.

### Priority 3: Separate `turn_time` from `event_time`

Construction should preserve both:

```text
turn_time: conversation timestamp
event_time: time of the described event
relative_time_text: original relative phrase
time_anchor: resolved anchor, if available
```

Retrieval and final answer synthesis should prefer `event_time` for "when did
X happen" questions, and use `turn_time` only as context for resolving relative
phrases.

### Priority 4: Add answer-detail preservation to final answer prompt

The final answer stage should avoid over-compressing evidence when the question
asks for attributes, reasons, lists, or descriptions.

Prompt rule:

```text
If the evidence contains specific answer-bearing details, preserve them.
Do not replace a detailed answer with a generic summary.
```

This targets cases like `Adjusting well` vs `learning commands and house training`.

### Priority 5: Add same-name event disambiguation in retrieval scoring

When multiple events share similar names or actions, rerank by full constraint
match:

```text
score = semantic_similarity
      + actor_match
      + time_match
      + object_match
      + purpose_match
      + source_turn_relevance
      - contradiction_penalty
```

This should be applied before both graph jump expansion and raw fallback evidence
selection.

### Priority 6: Treat Cat5 as a separate mode

Cat5 should not use the same "find something related and answer" behavior as
Cat1-Cat4. A useful rule:

```text
If the question contains a specific constraint and retrieved evidence only
matches the general topic, refuse.
```

Cat5-specific diagnostics should track:

- false positive answer rate
- near-miss evidence rate
- missing-constraint reasons
- whether refusal came from no evidence or insufficient evidence

---

## 7. Suggested Coding Plan

### Stage A: Evidence verifier

Files likely involved:

```text
src/graphmemory/graph_retrieval.py
src/graphmemory/prompts.py
tests/
```

Add structured evidence verification before final answer. Start with a prompt
based verifier, then add deterministic checks for actor/time/object when fields
exist.

Commit after tests.

### Stage B: Event slot schema

Files likely involved:

```text
src/graphmemory/graph_build.py
src/graphmemory/prompts.py
src/graphmemory/graph_store.py
tests/
```

Extend construction prompt and repair logic to preserve event slots. Keep this
backward-compatible with existing graph JSON.

Commit after tests.

### Stage C: Constraint-aware reranking

Files likely involved:

```text
src/graphmemory/graph_retrieval.py
tests/
```

Use the event slots to rerank graph candidates and raw fallback hits. Add tests
for same-action events with different actors/purposes.

Commit after tests.

### Stage D: Time handling

Files likely involved:

```text
src/graphmemory/graph_build.py
src/graphmemory/graph_retrieval.py
src/graphmemory/prompts.py
tests/
```

Separate `turn_time`, `event_time`, and `relative_time_text`. Add tests where
conversation date differs from the described event date.

Commit after tests.

### Stage E: Re-run focused evaluation

Recommended evaluation order:

1. Run unit tests.
2. Run targeted QA on known problematic examples:
   - 5K charity run actor/cause questions.
   - Cat5 near-miss food/activity questions.
   - event date vs session date questions.
3. Run full `conv-41` QA + LLM judge.
4. Compare:
   - overall accuracy
   - Cat5 accuracy
   - forced finish rate
   - raw fallback rate
   - same-event confusion examples

---

## 8. Bottom Line

The current branch made retrieval more robust and observable, but the remaining
errors show that the next bottleneck is evidence validation rather than simple
graph expansion.

The highest-impact next step is:

```text
Add evidence sufficiency verification before final answer,
then strengthen event signatures so retrieval can disambiguate similar events.
```

This should improve Cat5 refusal and reduce same-event contamination without
undoing the recall gains from raw fallback and multi-seed localization.
