"""Trained-head-only stochastic inference, with normalization and stochastic depth frozen."""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from trustretina.evaluation.metrics import aggregate_patient_mean


def stochastic_inventory(model):
    return {'dropout':[{'name':n,'type':type(m).__name__,'p':m.p} for n,m in model.named_modules() if isinstance(m,nn.modules.dropout._DropoutNd)],'stochastic_depth':[{'name':n,'p':getattr(m,'p',None)} for n,m in model.named_modules() if type(m).__name__=='StochasticDepth'],'normalization':sorted({type(m).__name__ for m in model.modules() if 'Norm' in type(m).__name__})}


@contextmanager
def trained_head_dropout(model):
    inventory=stochastic_inventory(model)
    active=[x for x in inventory['dropout'] if x['p']>0]
    if active!=[{'name':'head.0','type':'Dropout','p':.2}]:raise ValueError('Expected trained head dropout unsupported; do not insert layers')
    flags={m:m.training for m in model.modules()};model.eval();model.head[0].train()
    try:yield
    finally:
        for module,flag in flags.items():module.training=flag


def head_patient_samples(model,features,rows,*,passes,base_seed):
    if passes!=20:raise ValueError('Phase 6 fixes twenty MC passes')
    devices=list(range(torch.cuda.device_count())) if features.is_cuda else []
    labels=np.asarray([r['labels'] for r in rows]);samples=[];ids=None
    with torch.random.fork_rng(devices=devices),trained_head_dropout(model),torch.inference_mode():
        for s in range(passes):
            torch.manual_seed(base_seed+s)
            if features.is_cuda:torch.cuda.manual_seed_all(base_seed+s)
            eye=model.head(features).sigmoid().cpu().numpy()
            patient=aggregate_patient_mean([r['patient_id'] for r in rows],[r['eye'] for r in rows],labels,eye,[r['component_id'] for r in rows])
            if ids is None:ids=patient['patient_ids']
            elif ids!=patient['patient_ids']:raise ValueError('MC patient order changed')
            samples.append(patient['probabilities'])
    return {'patient_ids':ids,'probabilities':np.stack(samples),'seeds':[base_seed+s for s in range(passes)],'passes':passes,'scope':'trained head.0 only; stochastic depth/normalization evaluation mode'}
