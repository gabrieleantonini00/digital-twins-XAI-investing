# Data dictionary: OpenAI 500-persona study

`Data_500.csv`: 500 unique personas, one assigned treatment and one investment response per row.

| Field | Meaning | Source / allowed values |
|---|---|---|
| pid | Original profile identifier | Unique Twin-2K-500 participant ID |
| Market_Context | Randomized market vignette | Bull Market (High Growth); Bear Market (Severe Downturn) |
| Transparency | Randomized disclosure vignette | Standard AI (Performance metrics only); Explainable AI (Detailed rationales) |
| Financial_Literacy | Baseline literacy proxy | Model-inferred integer 1–7, before scenario exposure; 4 may indicate intermediate knowledge or insufficient evidence; not a validated human test score |
| Tech_Savviness | Baseline digital-familiarity proxy | Model-inferred integer 1–7, before scenario exposure; not a validated human test score |
| Age | Explicitly stated age | Integer 18–120 or missing; no age inferred from an approximate age category |
| Gender | Explicitly stated gender | Text or missing; not inferred from names or stereotypes |
| Education | Explicitly stated education | Text or missing |
| Willingness_to_Invest | Single simulated response | Integer 1 (Absolutely not willing) through 7 (Extremely willing) |

Missing demographic values are intentional and do not exclude a persona from the
primary analysis. All 500 baseline proxy scores and all 500 investment responses
must be valid before dataset export. Earlier Llama/Groq pilot outcomes are stored
separately and never appended to this file.
