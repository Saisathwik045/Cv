"""Frozen research candidate. No supported automatic acceptance policy exists in v4."""
import numpy as np
from trustretina.config import CLASSES
from .development import temperature_probabilities, signals


def candidate_outputs(patient_probabilities,patient_passes,quality,*,temperature):
    calibrated=temperature_probabilities(patient_probabilities,temperature)
    samples=np.stack([temperature_probabilities(p,temperature) for p in patient_passes])
    s=signals(calibrated,samples,quality)
    return {'probabilities':calibrated,'positive_labels':[[label for label,value in zip(CLASSES,row) if value>=.5] for row in calibrated],'explanation_target_indices':np.asarray(patient_probabilities).argmax(1),'confidence':s['confidence'],'uncertainty':s['uncertainty'],'quality':s['quality'],'decision':['REVIEW REQUIRED']*len(calibrated),'policy_fitted':False}
