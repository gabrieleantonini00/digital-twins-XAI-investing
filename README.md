# Explainable AI and simulated investment intentions

Experiment with **500 distinct personas**, randomly assigned to Market
(Bull/Bear) x Transparency (Standard/XAI): **125 profiles per condition** and
one investment response per persona. OpenAI `gpt-5.6-sol`, reasoning `medium`,
temperature **1**. No temperature comparison or repeated outcome sampling.

## Results

The proposed three-way interaction was not statistically detectable:
likelihood-ratio chi-square = 0.897, df = 1, **p = 0.344**. OLS-HC3 and ART
sensitivity checks also did not detect it (p = 0.498 and 0.355).

| Market | Standard AI mean | XAI mean |
|---|---:|---:|
| Bear | 2.39 | 3.15 |
| Bull | 3.98 | 4.97 |

All six planned ordinal-model XAI contrasts were positive and significant after
Holm correction. Proportional-odds diagnostics flagged literacy-related
departures, so common odds ratios require caution. Financial literacy is an
LLM-inferred baseline proxy, not a validated human test score. These simulated
outcomes do not establish human investment behavior or algorithm aversion
relative to human advice. See the manuscript for uncertainty and limitations.

Estimated Batch API charge: **USD 8.88**, excluding the technical compatibility
check and taxes. The completed-run audit is consolidated in `PROTOCOL.md`.

## Files

- `Data_500.csv`, `Data_Dictionary.md`: final sample and variable definitions.
- `simulation.py`: fixed configuration, sampling, prompts and resumable Batch collection.
- `Analysis.R`: all statistical models and checks; writes one report and two figures.
- `PROTOCOL.md`: prospective analysis plan and completed-run audit.
- `paper/main.tex`, `paper/Manuscript.pdf`: one self-contained LaTeX source and its PDF.
- `results/analysis.txt`: descriptive tables, estimates, diagnostics, sensitivity tests and software versions.
- `results/*.png`: the two manuscript figures.
- `tests/test_simulation.py`: offline pipeline checks.

The earlier 100-persona study remains in the repository's `pilot-100` tag and Git
history. Its observations are not pooled with the current sample. There is no
paper-generation script or public JSON metadata to maintain.

## Reproduce the analysis and PDF

Use Python 3.10+ and R with packages `ordinal`, `ARTool` and `ggplot2`.

```sh
python -m pip install -r Requirements.txt
Rscript Analysis.R
python -m unittest discover -s tests -v
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Manuscript main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Manuscript main.tex
```

The manuscript tables and prose are stored directly in `main.tex`. Re-running
the analysis regenerates the report and figures; changes to numerical findings
must be reflected in the LaTeX source before recompiling.

To collect a new run, supply credentials through the environment or a local
`.env` based on `.env.example`, then use `python simulation.py prepare` and
`python simulation.py run`. The fixed configuration is at the top of the script.
The second command submits paid requests. Private source profiles, request
files, raw responses and resumable JSON checkpoints stay in ignored `runs/`.
The $30 planning ceiling is not a provider-enforced account spend limit.

## Source attribution

Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025).
[Twin-2K-500](https://arxiv.org/abs/2505.17479), source data CC BY 4.0.
The exact source revision, seed, parameter settings and audit are recorded in
the simulation script and protocol.
