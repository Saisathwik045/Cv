"""Deterministic patient-predictor Grad-CAM; no parameter updates or localization claims."""
import torch
import numpy as np
from torch.nn import functional as F


def patient_gradcam(model,images,*,layer=None,target_index=None):
    if images.ndim!=4 or len(images)!=2:raise ValueError('Exactly both eye images of one patient required')
    model.eval();target_layer=layer if layer is not None else model.backbone.features[-1]
    saved=[]
    hook=target_layer.register_forward_hook(lambda module,inputs,output:saved.append(output))
    try:
        x=images.detach().clone().requires_grad_(True)
        logits=model(x);p=logits.sigmoid().mean(0);target=int(p.detach().argmax()) if target_index is None else target_index
        if not isinstance(target,int) or not 0<=target<8:raise ValueError('Explanation target must be a class index 0..7')
        if len(saved)!=1 or saved[0].ndim!=4:raise ValueError('Require one spatial BCHW activation')
        activation=saved[0];grad=torch.autograd.grad(p[target],activation,retain_graph=False)[0]
        cam=(grad.mean((2,3),keepdim=True)*activation).sum(1,keepdim=True).relu()
        cam=F.interpolate(cam,size=images.shape[-2:],mode='bilinear',align_corners=False)[:,0]
        maximum=cam.flatten(1).amax(1)[:,None,None]
        normalized=torch.where(maximum>0,cam/maximum.clamp_min(1e-30),torch.zeros_like(cam))
        return {'heatmaps':normalized.detach().cpu().numpy(),'patient_probabilities':p.detach().cpu().numpy(),'target_index':target,'positive_indices':torch.nonzero(p.detach()>=.5).flatten().cpu().tolist(),'target_below_threshold':bool(p[target].detach()<.5),'zero_maps':(maximum.flatten()==0).detach().cpu().tolist(),'score':'mean of both eye sigmoid probabilities for target','interpretation':'explanation, not validated lesion localization'}
    finally:hook.remove()
