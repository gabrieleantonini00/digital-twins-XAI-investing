"""Generate an artificial fixture solely for offline software verification."""
from pathlib import Path
import random
import pandas as pd
root = Path(__file__).resolve().parent
rng = random.Random(104)
rows = []
for market in ['Bear Market (Severe Downturn)', 'Bull Market (High Growth)']:
    for transparency in ['Standard AI (Performance metrics only)', 'Explainable AI (Detailed rationales)']:
        for _ in range(125):
            lit = rng.randint(1, 7)
            u = rng.uniform(.0001, .9999)
            import math
            latent = .6 * ('Explainable' in transparency) + .1*(lit-4) + math.log(u/(1-u))
            score = 1 + sum(latent > cut for cut in [-2, -1.2, -.5, .3, 1.2, 2.1])
            rows.append(dict(pid=len(rows)+1, Market_Context=market, Transparency=transparency,
                             Financial_Literacy=lit, Willingness_to_Invest=score))
path = root / 'runs/test-analysis.csv'
pd.DataFrame(rows).to_csv(path, index=False)
print('Artificial software fixture saved in runs/; excluded from study outputs.')
