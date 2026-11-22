"""Self-contained, in-memory demonstration export. No files, external URLs or scripts."""
import base64
import html
import io
import json
from datetime import datetime, timezone

from dashboard.presentation import SIGNALS, WORKFLOW, label_name


def _image_uri(image):
    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    image.thumbnail((1000, 1000))
    image.info.clear()
    data = io.BytesIO()
    image.save(data, format='PNG')  # No source filenames or EXIF metadata in export.
    return 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode('ascii')


def demonstration_report(result, inputs, overlays, target, *, zero_maps=(False, False)):
    """Include only the current pair/result and the currently selected target's maps."""
    if len(inputs) != 2 or len(overlays) != 2 or len(zero_maps) != 2 or target not in result['class_order']:
        raise ValueError('A complete current pair and two current target overlays are required')
    esc = lambda value: html.escape(str(value), quote=True)
    figures = []
    for heading, images in [('Uploaded input pair', inputs),
                            (f'Grad-CAM · {label_name(target)}', overlays)]:
        flags = zero_maps if images is overlays else (False, False)
        figures.append(f'<section><h2>{esc(heading)}</h2><div class="pair">' + ''.join(
            f'<figure><img src="{_image_uri(im)}" alt="{eye} eye"><figcaption>{eye} eye{" · zero activation map" if zero else ""}</figcaption></figure>'
            for im, eye, zero in zip(images, ('Left', 'Right'), flags)) + '</div></section>')
    rows, bars = [], []
    for i, label in enumerate(result['class_order']):
        raw, cal = result['raw_probabilities'][label], result['calibrated_probabilities'][label]
        rows.append(f'<tr><td>{esc(label_name(label))}</td><td>{raw:.4f}</td><td>{cal:.4f}</td><td>{"Above threshold" if cal >= result.get("threshold", .5) else "Below threshold"}</td></tr>')
        y = 28 + i * 44
        bars.append(f'<text x="4" y="{y+12}">{esc(label)}</text><rect x="40" y="{y}" width="{raw*500:.3f}" height="12" fill="#a3c7c0"/><rect x="40" y="{y+15}" width="{cal*500:.3f}" height="12" fill="#176b60"/>')
    threshold_x = 40 + result.get('threshold', .5) * 500
    chart = '<svg viewBox="0 0 600 410" role="img" aria-label="Raw and calibrated label probabilities">' + ''.join(bars) + f'<path d="M{threshold_x} 15 V385" stroke="#9b5b20" stroke-dasharray="5 4"/><text x="40" y="405">0%</text><text x="510" y="405">100%</text></svg>'
    positive = ', '.join(label_name(k) for k in result['positive_labels']) or 'None at the fixed threshold; this does not establish a healthy retina.'
    signals = ''.join(f'<p><strong>{esc(name.title())}: {result["signals"]["calibrated"][name]:.6f}</strong><br>{esc(detail)}</p>' for name, detail in SIGNALS.items())
    metadata = {k: result.get(k, 'Not recorded') for k in ('model', 'best_epoch', 'device', 'temperature', 'threshold', 'mc_passes', 'inference_seconds', 'contract_sha256', 'checkpoint_sha256', 'preprocessing_sha256', 'temperature_sha256', 'quality_normalizer_sha256')}
    metadata['input_display'] = 'EXIF-oriented RGB previews, resized only for this report; model preprocessing is 224 px letterbox'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>TrustRetina demonstration report</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#162f35;max-width:960px;margin:32px auto;padding:0 20px}}h1,h2{{line-height:1.2}}h2{{margin-top:28px}}.notice{{background:#fff8db;padding:14px;border-radius:8px}}.pair{{display:flex;gap:20px}}figure{{margin:0;width:50%;text-align:center}}img{{max-width:100%;height:auto;border-radius:8px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #d4e6e1;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}svg{{width:100%;max-height:450px}}section,figure{{break-inside:avoid}}@page{{size:A4;margin:14mm}}@media print{{body{{margin:0;padding:0;font-size:11pt}}h1{{font-size:23pt}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}</style></head><body>
<h1>TrustRetina · demonstration report</h1><p>Exported {esc(datetime.now(timezone.utc).isoformat())} (UTC). Open this file locally and use Print → Save as PDF for a complete printable copy.</p>
<p class="notice"><strong>REVIEW REQUIRED · Research demonstration, not a diagnosis.</strong> No validated automatic acceptance policy is fitted. This report contains the uploaded images and outputs; keep it private. Resetting the app cannot revoke downloaded copies.</p>
<h2>Labels above threshold</h2><p>{esc(positive)}</p><p>Independent patient-derived research labels; confidence and class probabilities are not accuracy scores.</p>
{figures[0]}<section><h2>Prediction probabilities</h2>{chart}<p>Light: raw. Dark: calibrated. Dashed line: threshold {result.get('threshold', .5)}. Probabilities need not sum to 100%.</p>
<table><thead><tr><th>Research label</th><th>Raw</th><th>Calibrated</th><th>Threshold result</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<h2>Reliability signals</h2>{signals}{figures[1]}<p>Grad-CAM describes target sensitivity on the model canvas, not verified lesion localization or proof of correctness. Explanation target: {esc(label_name(target))}; it is separate from the complete positive-label list.</p>
<h2>Method</h2><p>{esc(WORKFLOW)}</p><p>The deployed research model uses 224 px preprocessing. No reference labels were supplied for this pair; no confusion matrix or accuracy is computed for it. Historical evaluation belongs on Research results. Clinical validity and full deployment readiness are not established by this export.</p>
<h2>Model and runtime record</h2><pre>{esc(json.dumps(metadata, indent=2, allow_nan=False))}</pre>
<h2>Complete exported prediction record</h2><pre>{esc(json.dumps(result, indent=2, allow_nan=False))}</pre></body></html>'''
