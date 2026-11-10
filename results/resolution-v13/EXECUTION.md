# V13 verified execution audit

All six completed; use the separately generated frozen paired comparison.

| Run | Status | Epochs / best | Macro AUROC | Macro AP | Macro F1 | Micro F1 |
|---|---|---:|---:|---:|---:|---:|
| fold0_res224 | complete | 13 / 7 | 0.8382533015958769 | 0.5728531961135227 | 0.5302441035998697 | 0.5432489451476793 |
| fold0_res320 | complete | 11 / 5 | 0.8531594073967494 | 0.5879198190284902 | 0.5699899817864165 | 0.5741758241758241 |
| fold1_res224 | complete | 13 / 7 | 0.8394975768353139 | 0.5893636137575324 | 0.5708497401819125 | 0.5427189682966147 |
| fold1_res320 | complete | 15 / 9 | 0.8522321212910886 | 0.6019087790957776 | 0.5760185281229697 | 0.574750830564784 |
| fold2_res224 | complete | 10 / 4 | 0.8204519882075404 | 0.5561450684150397 | 0.5116593095167377 | 0.5200411099691675 |
| fold2_res320 | complete | 12 / 6 | 0.8478475166830179 | 0.5925237828549474 | 0.5686741889857789 | 0.5763358778625954 |

Source/nonce/output hashes checked; embedded worker statically decoded without execution and matched approved bytes. Fit-only ratio weights, inner-only checkpoint selection, frozen outer labels/groups and mean-eye probabilities are verified for completed outputs. Patient-level artifacts stay private. No model promotion, final evaluation, calibration, threshold fitting or all-data refit.
