# All-eight-class development comparison

Recorded V14/V15 reused-validation aggregates only. This is not an independent
validation leaderboard or a new scientific computation. Both frozen candidates
use the same fixed p>=0.5 decisions before/after shared-temperature calibration.
The live demo remains a different, original v5/Phase6 candidate.

Source: [public aggregate JSON](../results/calibration-v15/aggregate-results.json)
and [V15 interpretation](../results/calibration-v15/RESULTS.md).
See the [research defence chapter](ACADEMIC_RESEARCH_DEFENCE.md).

Positive/negative support counts patients, not independent eyes. Precision,
recall, F1, FP and FN are copied from recorded classification aggregates. ECE
values are copied from the raw and calibrated class reliability aggregates.
Missing/undefined values are not replaced with zero. Values below are rounded
for presentation; source JSON preserves full precision.

| Class | px | Positive | Negative | Precision | Recall | F1 | FP | FN | Raw ECE | Calibrated ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| N | 320 | 175 | 350 | 0.511551 | 0.885714 | 0.648536 | 148 | 20 | 0.205686 | 0.202181 |
| N | 224 | 175 | 350 | 0.461078 | 0.880000 | 0.605108 | 180 | 21 | 0.257786 | 0.257643 |
| D | 320 | 169 | 356 | 0.756098 | 0.366864 | 0.494024 | 20 | 107 | 0.066149 | 0.048581 |
| D | 224 | 169 | 356 | 0.722222 | 0.230769 | 0.349776 | 15 | 130 | 0.080528 | 0.080118 |
| G | 320 | 32 | 493 | 0.592593 | 0.500000 | 0.542373 | 11 | 16 | 0.033078 | 0.042682 |
| G | 224 | 32 | 493 | 0.450000 | 0.562500 | 0.500000 | 22 | 14 | 0.057149 | 0.057331 |
| C | 320 | 32 | 493 | 0.238462 | 0.968750 | 0.382716 | 99 | 1 | 0.239213 | 0.247987 |
| C | 224 | 32 | 493 | 0.482759 | 0.875000 | 0.622222 | 30 | 4 | 0.091585 | 0.091804 |
| A | 320 | 25 | 500 | 0.687500 | 0.440000 | 0.536585 | 5 | 14 | 0.026815 | 0.033326 |
| A | 224 | 25 | 500 | 0.545455 | 0.720000 | 0.620690 | 15 | 7 | 0.058964 | 0.059152 |
| H | 320 | 16 | 509 | 0.400000 | 0.250000 | 0.307692 | 6 | 12 | 0.052593 | 0.062876 |
| H | 224 | 16 | 509 | 0.192308 | 0.312500 | 0.238095 | 21 | 11 | 0.097430 | 0.097732 |
| M | 320 | 26 | 499 | 0.880000 | 0.846154 | 0.862745 | 3 | 4 | 0.019788 | 0.025048 |
| M | 224 | 26 | 499 | 0.875000 | 0.807692 | 0.840000 | 3 | 5 | 0.022442 | 0.022593 |
| O | 320 | 147 | 378 | 0.441315 | 0.639456 | 0.522222 | 119 | 53 | 0.179954 | 0.183187 |
| O | 224 | 147 | 378 | 0.421569 | 0.585034 | 0.490028 | 118 | 61 | 0.179633 | 0.179717 |

## What to discuss

- C:320 produces many more false positives, with FN1 versus FN4 at224. Inspect FP/FN and F1 together.
- A/G:320 has lower recall than224; a favorable aggregate macro AUROC does not
  establish improvement for every class or at the deployed decision rule.
- H: only16 positive validation patients; reduced FP coexists with more FN.
- O: heterogeneous labels complicate interpretation; no clinical relabeling is justified.
- N/D validation ECE decreases in both models, while G/C/A/H/M/O ECE increases.
- Temperatures preserve all binary decisions. These precision/recall/F1 changes
  between resolutions are model differences, not benefits produced by calibration.

No resolution is declared an independent or clinical winner. All decisions remain
**REVIEW REQUIRED**. Suitable external ground truth and prospective independent
confirmation remain incomplete.
