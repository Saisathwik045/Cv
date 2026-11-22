# Verify the existing dashboard

Codespaces is an optional development environment. It is not part of the trained
model, prediction algorithm or required dashboard workflow. Do not create another
app or migrate models merely to open the existing Streamlit app.

The **Cloud readiness** page remains useful: it reports environment failures and
does not load the checkpoint or evaluate model accuracy. Removing it would not
fix those failures. The private model setup screen is needed only when the
original bundle is absent; it is not a page to repeat after a successful setup.

## A short demonstration

1. Open the existing app from Streamlit **My apps** under `Saisathwik045`.
   Keep the app private. Do not change budgets, recreate the deployment or paste
   credentials into chat.
2. Open **Guided synthetic demo** and run the CPU demonstration. Confirm eight
   label probabilities and the unchanged **REVIEW REQUIRED** decision. These
   generated images test software operation and have no medical meaning.
3. Generate the selected target's Grad-CAM for both eyes. Confirm that both
   overlays render; their appearance does not verify lesion localization.
4. Export the result JSON. Replace an input and confirm stale predictions and
   explanations disappear; reset the session and confirm results clear.
5. Open **Research study & viva** for recorded research findings. Keep these
   aggregate evaluation results separate from a single demonstration prediction.

For manual uploads, use the original matching left/right files for one eligible
development/demo patient. Do not use screenshots, unrelated eye pairs, or the
historical final cohort. Ten example pairs test interaction across inputs; they
do not measure accuracy without authoritative labels and an approved evaluation
protocol. Different images can legitimately produce the same positive label.

## What can be concluded

| Evidence | Supports | Does not establish |
|---|---|---|
| Result JSON | Recorded model identities, eight outputs, calibration arithmetic, threshold decisions and uncertainty fields | Independent re-execution, medical correctness or dataset accuracy |
| Paired Grad-CAM screenshot | The interface produced both explanation overlays | Correct lesions, causality or diagnostic validity |
| Readiness snapshot | Observed hashes, dependency versions and local configuration checks at that time | Platform authentication, external access, inference or clinical validation |
| Passing synthetic software tests | Tested code paths work under the tested conditions | Real-data performance or hosted security |

Do not present confidence or a calibrated class probability as model accuracy.
An exported checkpoint hash is a recorded identity; local hash verification and
an independently observed execution are stronger, separate evidence.

## Full readiness remains blocked on Community Cloud

The official [Community Cloud limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/status#configuration)
override `server.enableXsrfProtection=false`. The project requires it true, so the
strict readiness result remains FAIL. Restarting, changing config.toml or hiding
the readiness page cannot solve that conflict. A functional research demonstration
must not be described as a fully verified deployment.

Exact dependency mismatches are also failures and must not be waived to obtain a
green result. A controlled private environment is a separate deployment option
only when a fully passing strict check is required. Its private access, packages,
server protections and actual execution must all be checked before claiming PASS.
Do not enable paid resources or transfer private artifacts automatically.
