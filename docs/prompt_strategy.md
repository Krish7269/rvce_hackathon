# Prompt Optimization Strategy

## Objectives
- Convert natural language analytics goals into deterministic Python scripts.
- Maintain <3s generation latency via caching + retry minimization.

## Prompt Anatomy
1. **Role Framing** – positions Gemini as senior analyst for authority.
2. **Context Blocks** – embed query intent, dataset summary, metadata (columns, types, stats, rows).
3. **Output Contract** – numbered rules covering imports, comments, CSV_PATH usage, save-to-file, visualization, `__RESULT__` JSON line.
4. **Analysis Expectations** – remind Gemini of pandas aggregates, seaborn styling, ambiguity handling.

## Reliability Controls
- Use deterministic wording (“Output ONLY valid Python code”).
- Require explicit figure saving + final JSON marker to simplify downstream parsing.
- Include commentary requirement so code stays interpretable.

## Iterative Refinement
- When validator fails, append corrective snippet referencing the error and retry count.
- Escalate language (“Guarantee syntactic correctness”) for later attempts.

## Latency Optimizations
- LRU cache keyed by normalized query + metadata.
- Keep prompt template static; only small refinement deltas between retries.


