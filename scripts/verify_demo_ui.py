#!/usr/bin/env python3
"""Local browser interaction on private development examples; no publication."""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'artifacts/demo-v8'
PUBLIC = ROOT / 'results/demo-v8'


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1080})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        started = time.perf_counter()
        page.goto('http://127.0.0.1:8501', wait_until='networkidle')
        page.get_by_role('heading', name='Two eyes. One patient-level prediction.').wait_for()
        page.screenshot(path=str(PUBLIC / 'preview-empty.png'), full_page=True)
        assert page.get_by_role('button', name='Analyze paired images').is_disabled()
        inputs = page.locator('input[type=file]')
        inputs.nth(0).set_input_files(str(PRIVATE / 'left.png'))
        assert page.get_by_role('button', name='Analyze paired images').is_disabled()
        inputs.nth(1).set_input_files(str(PRIVATE / 'right.png'))
        page.get_by_role('button', name='Analyze paired images').click()
        page.get_by_role('heading', name='All predicted-positive labels').wait_for(timeout=60000)
        with page.expect_download() as d:
            page.get_by_role('button', name='Download research result JSON').click()
        download = d.value
        download.save_as(str(PRIVATE / 'ui-result.json'))
        result = json.loads((PRIVATE / 'ui-result.json').read_text())
        assert result['decision'] == 'REVIEW REQUIRED' and len(result['raw_probabilities']) == 8 and result['mc_passes'] == 20
        page.get_by_role('combobox').click()
        page.get_by_role('option', name='D', exact=True).click()
        page.get_by_role('button', name='Generate per-eye Grad-CAM').click()
        page.get_by_text('Left eye · D', exact=True).wait_for(timeout=60000)
        page.get_by_text('Right eye · D', exact=True).wait_for()
        page.screenshot(path=str(PRIVATE / 'paired-development-preview.png'), full_page=True)
        page.get_by_role('button', name='Reset patient session').click()
        page.get_by_role('heading', name='Two eyes. One patient-level prediction.').wait_for()
        page.get_by_role('heading', name='All predicted-positive labels').wait_for(state='hidden')
        assert page.get_by_role('button', name='Analyze paired images').is_disabled()
        assert page.get_by_text('left.png', exact=True).count() == 0
        assert page.locator('[data-testid="stImage"]').count() == 0
        page.locator('input[type=file]').nth(0).set_input_files({'name': 'invalid.png', 'mimeType': 'image/png', 'buffer': b'not an image'})
        page.get_by_text('This image cannot be decoded. Upload a valid JPEG or PNG.', exact=True).wait_for()
        assert page.get_by_role('button', name='Analyze paired images').is_disabled()
        page.get_by_role('button', name='Reset patient session').click()
        page.get_by_text('Research results', exact=True).click()
        page.get_by_role('heading', name='Five-backbone validation comparison').wait_for()
        page.get_by_role('heading', name='Frozen held-out evaluation · 350 patients').wait_for()
        page.screenshot(path=str(PUBLIC / 'preview-results.png'), full_page=True)
        assert not errors, errors
        report = {'browser': 'system Chromium / Playwright 1.56.0', 'url': 'http://127.0.0.1:8501', 'paired_interaction': 'private development pair from third prespecified validation patient', 'single_eye_blocked': True, 'json_download_verified': True, 'eight_probabilities': True, 'review_required': True, 'selectable_d_gradcam_both_eyes': True, 'reset_clears_uploads_predictions_explanations': True, 'invalid_image_error': True, 'results_page_separate_tables': True, 'browser_errors': errors, 'seconds': time.perf_counter() - started, 'screenshots': {'public': ['preview-empty.png', 'preview-results.png'], 'private_development': 'artifacts/demo-v8/paired-development-preview.png'}}
        (PUBLIC / 'ui-verification.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == '__main__':
    main()
