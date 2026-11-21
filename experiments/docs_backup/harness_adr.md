# Architecture Decision Record — development harness, provider, and probe execution

Status: accepted. Date: 2026-11-21.

## Decision 1: paper and ranking claims

Narrowed to a **preregistered replication-and-extension** (see `docs/amendment_1.md`).
arXiv:2604.20911 established benign-context decay of *formatting* prohibitions and is the
closest prior work; we do not claim to have discovered the phenomenon. This decision is
recorded before the main run.

## Decision 2: inference provider — BharatCode Chat Completions, streamed

Live `/models` (2026-11-21) returned three records. Selected by metadata, not assumption:

| id | status | context | max output | notes |
|---|---|---|---|---|
| glm-5.3-flash | live | 262,144 | 32,000 | primary actor model |
| deepseek-v4.1-flash | live | 1,000,000 | 32,000 | second actor model (Pro tier) |
| qwen-3.8-27b | live, **deprecated** | 262,144 | 32,000 | compat alias, sunset 2026-11-22T18:30Z — not used |

Measured behaviour that shaped the client (`src/providers/base.py`):

- **Queueing dominates latency.** A 2-token completion took 94–469 s wall clock. The server
  streams `: bharatcode queued` keep-alives, so requests are made with `stream=true`;
  a non-streaming call can sit silent past any reasonable read timeout.
- **Concurrency is free.** 12 simultaneous requests completed in 351 s total, with the same
  per-request latency distribution as serial calls. The queue is throughput-limited, not
  serialised. This justifies *branch-parallel probing*: one history prefix, nine probes in
  flight, instead of nine sequential replays of the same long prefix.
- **Validation is instant.** 400-class errors return in ~0.13 s, so malformed requests fail
  fast and are not retried.
- **`stream_options.include_usage`, `n`, `seed`, and native `tools` are accepted.**
  The `reasoning` (nested) field is rejected; top-level `reasoning_effort` is accepted.
  We pass `reasoning_effort: "none"` for short deterministic answers (measured ~95 s vs
  ~11 s to first token per the provider config notes).
- **Incomplete responses.** `finish_reason == "length"` (or a missing finish reason) marks a
  completion incomplete: the partial text is kept, flagged in the record, and its tool calls
  are not executed. No automatic retry at the same token budget.

Retry policy: exponential backoff (5, 10, 20 s + jitter) on 429/5xx/network; no retry on
4xx request-shape errors; `x-request-id` captured on every record. Requests have a
user-configurable `max_wait_s` cap — the BharatCode client will wait for the queue by design
but never indefinitely.

## Decision 3: development harness — direct Python (Approach A)

The assignment asks for a comparison of (A) direct Python, (B) OpenCode, (C) Mini-SWE-agent,
with a practical test. Result:

| Criterion | A: direct Python | B: OpenCode v2.3.2 (installed) | C: Mini-SWE-agent v2.4.6 (installed) |
|---|---|---|---|
| Setup | one `uv venv`, no service | already configured with BharatCode provider in `~/.config/opencode/opencode.jsonc` | installed via `uv tool` |
| API compatibility | verified live against `/models` and `/chat/completions` | verified: OpenCode's config points at the same base URL, OpenAI-compatible package | not needed |
| Determinism | full: every message, tool call, token count and seed is explicit | not deterministic (agent loop, context compaction, tool policy) | not deterministic |
| Cost per experiment | exactly the probe calls issued | extra calls for the coding loop itself | same |
| Reproducibility | JSONL of every request/response | session logs only | trajectories, but harness-specific |
| Suitability as *the experimental agent* | yes | **no** | **no** |

**Decision.** Experiments run on **A (direct Python)**. The experimental agent must be a
deterministic loop the researcher controls message-by-message; the assignment's integrity
rule ("separate the development agent from the experimental agent") rules out letting a
coding harness *be* the subject. B and C were used only as development tools while writing
this repository, never as the measured agent. No harness dependency is in
`requirements.txt`.

## Decision 4: probe execution — branch-parallel, evaluation turns never appended

Each checkpoint builds a history from the seeded sequence, then runs all eleven probes as
branches off that same prefix. Consequences:

- the probes are independent forward passes (identical to a sequential implementation);
- evaluation answers never enter the accumulation, so checkpoint *t* cannot be contaminated
  by the answers at *t−1* (the "auto-reinforcement confound" left open by 2604.20911);
- the history is reconstructed deterministically from `(group, n_items, seed)`, so any run
  can be reproduced from the manifest alone.

## Decision 5: methodological correction during the pilot

P11 was initially specified as a request to delete a file the user names explicitly. The
pilot's first checkpoint showed the agent correctly executing it (naming the file is exactly
what FS-SAFE-2 permits), so the probe measured compliance, not violation. It was redefined
as an **AUTH-5** probe (a named file with no approval record; the policy requires either a
naming request *or* an approval, and the policy's privileged-action clause governs),
the schema version was bumped to `probes-v2`, and every pilot record produced under v2
before the change was moved to `experiments/pilot/aborted_v2probe_superseded.jsonl` and
excluded from analysis. The loader refuses to mix schemas.
