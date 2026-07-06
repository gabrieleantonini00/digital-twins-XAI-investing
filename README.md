# digital-twins-XAI-investing
Experimental study on how Explainable AI (XAI) mitigates investor algorithm aversion during bear and bull markets using LLM digital twins.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the replication code, synthetic data generating process, and the final manuscript for the experimental study on Explainable AI (XAI) and algorithm aversion in financial markets.

## 📌 Project Overview
While the financial industry increasingly relies on automated algorithms, many investment funds keep their AI models opaque to avoid triggering **algorithm aversion** among retail investors. This study shifts the discourse from a generalized view of algorithmic trust toward a contingency-based framework. 

We investigate whether Explainable AI (XAI) can mitigate this aversion, explicitly testing how algorithmic transparency interacts with **macroeconomic market contexts** (Bull vs. Bear markets) and individual **financial literacy** levels.

### Key Findings
* **Significant 3-Way Interaction:** The effect of XAI is highly conditional (Market Context $\times$ Transparency $\times$ Financial Literacy).
* **Bear Markets:** XAI significantly mitigates algorithm aversion, but exclusively for individuals with low financial literacy, acting as an essential analytical scaffolding.
* **Bull Markets:** XAI generates a significant trust premium primarily among highly literate investors.

## 🔬 Experimental Design & Econometric Methodology
The study employs a computational "silicon sampling" methodology to conduct a **$2\times2\times2$ between-subjects factorial design**:
1. **Sample:** 100 Digital Twins instantiated via Large Language Models (LLMs), initialized using real socio-demographic backstories.
2. **LLM Engine:** Groq API (Llama-3.1-8b-instant) deployed with a strictly deterministic parameter configuration (Temperature = 0.0) to simulate cognitive biases.
3. **Statistical Inference:** Given the ordinal nature of the dependent variable (1-7 Likert scale) and significant departures from normality, hypothesis testing was conducted using a robust non-parametric **Aligned Rank Transform (ART) ANOVA**.

## 📂 Repository Structure
```text
├── data/                  # Simulated behavioral responses and demographic inputs
├── simulation.py                   # Python scripts for LLM API interaction and data generation
├── analysis.R              # Scripts for the ART ANOVA and post-hoc pairwise comparisons
├── paper/                 # LaTeX source code and the compiled PDF manuscript
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
