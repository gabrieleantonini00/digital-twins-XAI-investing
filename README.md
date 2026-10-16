# Explainable AI and simulated investment intentions

Replication project for a balanced experiment with **500 distinct personas**:
Market (Bull/Bear) Ã— Transparency (Standard/XAI), **125 profiles per condition**.
Each persona contributes exactly one willingness-to-invest response. There is no
temperature comparison, repeated outcome sampling, or pooling with the old pilot.

## Study configuration

- OpenAI `gpt-5.6-sol`, reasoning `medium`, temperature **1**.
- Source: [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), `full_persona`, pinned revision in `study_config.json`.
- Sample without replacement and allocation shuffle: seed 42.
- Two phases: scenario-blind baseline assessment, then one assigned investment vignette.
- Financial literacy is an **LLM-inferred pre-treatment proxy**, not a human test score.
- Batch transport: sequential lots of 200, 200 and 100 in each phase, respecting the account's queued-token limit.
- All requests use the same configuration. The scripts preserve checkpoints by PID and do not automatically redraw failed investment outcomes.

The outcome is willingness to invest (integer 1â€“7). The study compares two
descriptions of algorithmic disclosure. It does not identify human investment
behavior or algorithm aversion relative to human advice, since there is no
human-advisor condition.

## Reproduce

Use Python 3.10+ and R with packages `ordinal`, `ARTool`, `ggplot2`, and `jsonlite`.
Install the Python dependencies with `python -m pip install -r Requirements.txt`.
Put credentials in the environment or a local `.env` based on `.env.example`.
Never commit credentials, downloaded persona profiles, raw API responses, or request files.

```sh
python simulation.py prepare
python -m unittest -v test_simulation.py
python simulation.py run
Rscript Analysis.R
python build_paper.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Manuscript main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Manuscript main.tex
```

`prepare` fixes the sample and estimates a conservative token budget without
sending generation requests. `run` submits paid batches and resumes existing
batch IDs after interruptions. `status` retrieves progress. A submission whose
response was lost is reconciled by batch metadata rather than blindly repeated.
The $30 planning ceiling includes input estimates, cache-write overhead and output
caps; it is not a provider-enforced account spend limit. The exact realized token
usage and estimated charge are recorded in `study_audit.json` once all results are collected.

The initial 500-request baseline batch exceeded the account's queued-token limit
and was rejected before processing. The six smaller transport batches do not
alter the research sample or protocol.

## Analysis

`PROTOCOL.md` records the plan fixed before new investment-outcome collection;
this is not a claim of public preregistration. The primary cumulative logit model
retains the numerical literacy proxy and tests the three-way interaction by
likelihood-ratio comparison. Six planned XAI contrasts use Holm correction.
OLS with HC3 errors and the earlier median-split ART approach are sensitivity
analyses. The report includes model diagnostics and flags extrapolated contrasts.

## Files

- `Data_500.csv`: completed OpenAI sample, one row per persona; generated only after all 500 outcomes validate.
- `Data.csv`: original 100-persona pilot, preserved separately.
- `simulation.py`, `study_config.json`: sampling, prompts, structured outputs, batch orchestration and auditing.
- `Analysis.R`, `results/`: reproducible statistical analysis and generated tables/figures.
- `build_paper.py`, `paper/`: manuscript source and PDF, populated from completed results.
- `legacy/`: original manuscript source and original Groq code for historical reference.
- `runs/`, `.env`, `.packages/`, `.Rlib/`: local-only inputs, checkpoints, credentials and dependencies, excluded from publication.

The new manuscript replaces the pilot's numerical results and conclusions with
the newly estimated findings. Changes across versions cannot be attributed to
sample size alone because the model, temperature, measurement and analysis also changed.

## Source attribution

Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025).
[Twin-2K-500](https://arxiv.org/abs/2505.17479). Source data: CC BY 4.0.
See the paper for full references and measurement limitations.

## Results of the completed run

All 500 investment responses validated. Estimated Batch API charge: **USD 8.88**, excluding the no-persona compatibility check and taxes.

The primary three-way interaction was not statistically detectable (likelihood-ratio chi-square = 0.897, df = 1, p = 0.344). The OLS-HC3 and ART sensitivity checks also did not detect it (p = 0.498 and 0.355). Thus, the earlier pilot conclusion of a significant three-way interaction is not reproduced in this run.

Mean willingness to invest was 2.39 (Standard) versus 3.15 (XAI) in bear markets, and 3.98 versus 4.97 in bull markets. All six planned ordinal-model XAI contrasts were positive and significant after Holm correction; proportional-odds diagnostics flagged literacy-related departures, so common odds ratios require caution. See the manuscript for uncertainty and limitations.
