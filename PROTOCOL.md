# OpenAI study: 500 distinct personas

Protocol fixed before collecting the new investment outcomes (16 October 2026).

- Source: Twin-2K-500, full_persona, revision `f883165a3026fde855dfd448e0cd16443ab257b6`; 2,058 source profiles.
- Sample: pandas sample without replacement, N=500, seed 42; unique PIDs and unique persona summaries.
- Randomization: 125 profiles in each Market (Bull/Bear) × Transparency (Standard/XAI) condition; seeded shuffle. Financial literacy is a measured/inferred moderator, not a randomized third treatment.
- Engine: `gpt-5.6-sol`, reasoning `medium`, temperature 1, low output verbosity, at most 1,536 output tokens per request including reasoning. No top_p override. Same settings for every profile.
- Baseline: one scenario-blind request estimates financial literacy and tech savviness (1–7) and extracts demographics from the summary. These scores are **LLM-inferred proxies**, not validated human literacy tests. Unknown demographics remain missing.
- Outcome: one independent request per persona and one assigned scenario; integer willingness to invest from 1 to 7. Baseline scores are fixed inputs to this request. No prior response history, tools, or external knowledge retrieval.
- No temperature comparison, repeated outcome sampling, or mixing with the previous Llama/Groq dataset.
- Batch: two sequential phases, each with 500 independent requests, transported in sequential lots of 200, 200 and 100 to respect the account's 1,350,000 enqueued-token limit. The initial 500-request job was rejected at validation with zero processed requests. Save raw responses privately, map results by custom_id/PID, retain one valid outcome per persona, and require all 500 valid results before final export. Failed/incomplete responses stop the pipeline for inspection; no automatic re-draws.
- Cost: batch prices checked 16 October 2026; $2/M uncached input, $0.20/M cached input, $10/M output. Cache writes are conservatively costed at 1.25× uncached input. Planning ceiling $30 across both stages; this is not an account-level hard limit. Record actual token usage separately. A technical compatibility request contained no persona and is outside the sample.

## Analysis fixed before outcome collection

The primary outcome is ordinal. Fit a cumulative logit model with Market × Transparency × numerical Financial_Literacy (centered at 4); no data-driven median split in the primary analysis. Test the three-way interaction by a likelihood-ratio comparison with the hierarchical model containing all two-way terms. Report coefficients, odds ratios, confidence intervals, convergence, and proportional-odds diagnostics. If the moderator has no variation or the model is not identifiable, report that failure rather than substituting a favorable model.

Planned comparisons: XAI vs Standard at financial-literacy proxy values 2, 4, and 6 in each market; Holm correction across these six contrasts. Flag contrasts that extrapolate beyond observed literacy ranges in a treatment cell.

Sensitivity analyses: (1) OLS on the numeric 1–7 outcome with the same continuous moderator and HC3 standard errors; (2) the legacy ART analysis after a median split, clearly labeled as sensitivity. Report the distribution and actual subgroup sizes; ties at the median may prevent balanced literacy groups. Do not use a normality test of the raw Likert scores to select the primary model.

The study estimates responses within this particular LLM simulation. It does not establish human investment behavior or algorithm aversion relative to a human advisor, because no human-advisor condition is present. Re-estimate all results; earlier significant findings are not targets to reproduce.
