"""Thread-safe paired-eye inference; unchanged frozen candidate, no fitting or storage."""
import io
import json
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from trustretina.models.backbones import create_backbone
from trustretina.preprocessing.shared import PreprocessConfig, preprocess, letterbox_rgb
from trustretina.reliability.development import canonical_quality, temperature_probabilities, signals
from trustretina.reliability.stochastic import head_patient_samples
from trustretina.runner.final_analysis import frozen_quality
from trustretina.runner.reliability_runtime import state_hash
from trustretina.gradcam.implementation import patient_gradcam
from .bundle import DEFAULT_BUNDLE, CONTRACT_SHA256, verify_bundle

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000
# PyTorch RNG fork is process-global; serialize across cached service instances too.
INFERENCE_LOCK = threading.RLock()


def decode_image(content):
    if not isinstance(content, bytes) or not content or len(content) > MAX_BYTES:
        raise ValueError('Upload a nonempty JPEG or PNG smaller than 20 MiB for each eye.')
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in ('JPEG', 'PNG') or image.width * image.height > MAX_PIXELS:
                raise ValueError('Use JPEG or PNG with at most 25 million pixels.')
            image.load()
            return ImageOps.exif_transpose(image).convert('RGB').copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('This image cannot be decoded. Upload a valid JPEG or PNG.') from error


class InferenceService:
    def __init__(self, bundle=DEFAULT_BUNDLE, *, device='auto'):
        started = time.perf_counter()
        self.contract = verify_bundle(bundle)
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU or CUDA execution is supported')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        self.config = PreprocessConfig()
        if self.config.fingerprint() != self.contract['preprocessing_sha256']:
            raise ValueError('Frozen preprocessing fingerprint changed')
        bundle = Path(bundle)
        temperature = json.loads((bundle / 'temperature.json').read_text())['temperature']
        if temperature != self.contract['temperature']:
            raise ValueError('Frozen temperature changed')
        self.temperature = temperature
        with np.load(bundle / 'quality-normalizer.npz', allow_pickle=False) as f:
            self.normalizer = f['sorted_training_feature_distributions'].copy()
        frozen_quality(self.normalizer, np.zeros((1, 3)))
        record = torch.load(bundle / 'best.pt', map_location='cpu', weights_only=True)
        if record['completed_epoch'] != self.contract['best_epoch'] - 1:
            raise ValueError('Selected checkpoint epoch mismatch')
        self.model = create_backbone(self.contract['selected_backbone'])
        self.model.load_state_dict(record['model'], strict=True)
        del record
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.state_sha256 = state_hash(self.model)
        self.load_seconds = time.perf_counter() - started

    def prepare(self, left, right):
        if left is None or right is None:
            raise ValueError('Both left and right eye images are required; one eye cannot stand in for both.')
        images = [decode_image(left), decode_image(right)]
        tensors = torch.stack([preprocess(x, self.config, role='inference')[0] for x in images])
        canvases = [np.asarray(letterbox_rgb(x, self.config)[0]) for x in images]
        return tensors.to(self.device), canvases

    def predict(self, left, right):
        with INFERENCE_LOCK:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            tensors, canvases = self.prepare(left, right)
            self.model.eval()
            with torch.inference_mode():
                features = self.model.backbone(tensors)
                eye = self.model.head(features).sigmoid().cpu().numpy()
            raw = eye.mean(0, keepdims=True)
            # No patient identifiers or real labels are needed by the app aggregator.
            rows = [{'patient_id': 'upload', 'eye': e, 'labels': [0] * 8, 'component_id': 'upload'} for e in ('left', 'right')]
            mc = head_patient_samples(self.model, features, rows, passes=20, base_seed=self.contract['mc_validation_seed'])
            samples = mc['probabilities']
            cal = temperature_probabilities(raw, self.temperature)
            cal_samples = np.stack([temperature_probabilities(p, self.temperature) for p in samples])
            eye_quality = frozen_quality(self.normalizer, np.stack([canonical_quality(c.transpose(2, 0, 1)) for c in canvases]))
            quality = np.asarray([eye_quality.min()])
            raw_signals, cal_signals = [signals(p, s, quality) for p, s in ((raw, samples), (cal, cal_samples))]
            if any(m.training for m in self.model.modules()):
                raise RuntimeError('Frozen evaluation modes changed')
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            classes = self.contract['class_order']
            return {
                'schema': 8, 'model': self.contract['selected_backbone'], 'best_epoch': self.contract['best_epoch'],
                'contract_sha256': CONTRACT_SHA256, 'checkpoint_sha256': self.contract['checkpoint_sha256'],
                'preprocessing_sha256': self.config.fingerprint(), 'temperature_sha256': self.contract['temperature_sha256'],
                'quality_normalizer_sha256': self.contract['quality_normalizer_sha256'], 'temperature': self.temperature,
                'aggregation': self.contract['deterministic_predictor'], 'class_order': classes, 'threshold': .5,
                'raw_probabilities': dict(zip(classes, raw[0].tolist())), 'calibrated_probabilities': dict(zip(classes, cal[0].tolist())),
                'positive_labels': [label for label, value in zip(classes, cal[0]) if value >= .5],
                'signals': {'raw': {k: float(v[0]) for k, v in raw_signals.items()}, 'calibrated': {k: float(v[0]) for k, v in cal_signals.items()}},
                'per_eye_quality': dict(zip(('left', 'right'), eye_quality.tolist())),
                'raw_mc_variance': samples.var(0, ddof=0)[0].tolist(), 'calibrated_mc_variance': cal_samples.var(0, ddof=0)[0].tolist(),
                'mc_seeds': mc['seeds'], 'mc_passes': 20, 'mc_scope': mc['scope'],
                'mc_batch_context': 'single left/right pair; seeded masks depend on batch shape and runtime; not identical to cohort-wide saved masks',
                'decision': 'REVIEW REQUIRED', 'automatic_acceptance_policy_fitted': False,
                'device': str(self.device), 'inference_seconds': time.perf_counter() - started,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                'dependencies': {'torch': str(torch.__version__), 'torchvision': __import__('torchvision').__version__, 'numpy': np.__version__, 'Pillow': Image.__version__},
                'limitations': ['Research demonstration; not a diagnosis', 'Patient-derived labels; not verified eye diagnoses', 'Head-only MC uncertainty; technical quality proxy', 'All patients require review; no clinical risk guarantee'],
            }

    def explain(self, left, right, target_index):
        with INFERENCE_LOCK:
            tensors, canvases = self.prepare(left, right)
            cam = patient_gradcam(self.model, tensors, target_index=target_index)
            overlays = []
            for canvas, heat in zip(canvases, cam['heatmaps']):
                color = np.stack((heat, np.zeros_like(heat), 1 - heat), axis=2) * 255
                overlays.append(np.clip(.65 * canvas + .35 * color, 0, 255).astype(np.uint8))
            return cam, overlays
