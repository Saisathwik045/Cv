# Verify the existing dashboard

Codespaces is an optional development environment. It is not part of the trained
model, prediction algorithm or required dashboard workflow. Do not create another
app or migrate models merely to open the existing Streamlit app.

The dashboard now contains only **Paired-eye prediction** and **Research results**.
The five supporting pages were removed from navigation and routing at the owner's
request. Historical diagnostic modules and records remain available in source;
removing their routes does not fix recorded dependency or security failures.
The private model setup screen is still needed when the original bundle is absent;
it is a startup requirement, not an additional dashboard workspace.

## A short demonstration

1. Open the existing app from Streamlit **My apps** under `Saisathwik045`.
   Keep the app private; do not recreate it or paste credentials into chat.
2. On **Paired-eye prediction**, upload original matching left/right files from
   one eligible development/demo patient and select **Analyze paired images**.
   Confirm eight probabilities and the unchanged **REVIEW REQUIRED** decision.
3. Generate the selected target's Grad-CAM for both eyes. Confirm both overlays
   render; their appearance does not verify lesion localization.
4. Download the research result JSON. Replace an input and confirm stale results
   disappear, then use **Reset patient session** to clear uploads and outputs.
5. Open **Research results** for recorded model comparisons, held-out evaluation,
   reliability figures and the downloadable research report. Keep those findings
   separate from a single demonstration prediction.

Use original files, not screenshots or unrelated eye pairs. Do not use the
historical final cohort for further testing. Ten example pairs test interactions;
they do not measure accuracy without authoritative labels and an approved
protocol. Different images can legitimately produce the same positive label.
Generated non-patient fixtures may test upload mechanics but have no medical
meaning. No new fitting or model download is part of this walkthrough.

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
