"""Bounded label-blind content review; screening ranks work, never proves identity."""
from functools import lru_cache
import hashlib
import numpy as np
from .photo_verification import POLICY,prepare_photo,verify_photos,public_photo_record

POLICY_V3={**POLICY,'version':'bounded-retinal-review-v3','size':512,'sift_features':1600,
 'feature_contrast_threshold':.01,'feature_input':'CLAHE grayscale retinal foreground',
 'coarse_size':32,'maximum_detail_pairs':1536,'maximum_seconds':1800,
 'material_coarse_correlation':.98,'material_aligned_ssim':.90,
 'scope':'Photographic copies under the declared transforms; no clinical or patient identity adjudication',
 'budget_rule':'Prior technical sample/bridges and rejected controls first; alternate independent foreground intensity/gradient rankings; bounded outside-dHash nearest neighbours',
 'threshold_basis':'Preserve v2 copy tolerances; increased feature resolution/contrast to address measured feature scarcity, not class support or model outcomes'}


def coarse_content(rgb):
    import cv2
    gray=cv2.cvtColor(cv2.resize(np.asarray(rgb),(192,192),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
    contours,_=cv2.findContours((gray>8).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros_like(gray)
    if contours:cv2.drawContours(mask,[max(contours,key=cv2.contourArea)],-1,1,-1)
    y,x=np.nonzero(mask)
    crop=gray[y.min():y.max()+1,x.min():x.max()+1] if len(x) else np.zeros((32,32),np.uint8)
    small=cv2.resize(crop,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    yy,xx=np.mgrid[:32,:32];inside=(xx-15.5)**2+(yy-15.5)**2<14**2
    gradient=cv2.Laplacian(cv2.GaussianBlur(small,(3,3),0),cv2.CV_32F)
    def normalized(values):
        values=values[inside].copy();values-=values.mean();norm=float(np.linalg.norm(values))
        return values/max(norm,1e-8)
    return {'intensity':normalized(small),'gradient':normalized(gradient),'foreground_fraction':float(mask.mean()),'stddev':float(small[inside].std())}


def prepare_detail(rgb,raw_sha256=None):
    import cv2
    p=prepare_photo(rgb,raw_sha256,policy=POLICY_V3)
    gray=np.rint(p['_gray']*255).astype(np.uint8)
    enhanced=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
    sift=cv2.SIFT_create(nfeatures=POLICY_V3['sift_features'],contrastThreshold=POLICY_V3['feature_contrast_threshold'])
    points,descriptors=sift.detectAndCompute(enhanced,(p['_mask']*255).astype(np.uint8))
    p['_points']=np.array([k.pt for k in points],np.float32).reshape(-1,2);p['_descriptors']=descriptors;p['features']=len(points)
    p['low_information']=p['foreground_box'] is None or p['foreground_fraction']<.2 or p['foreground_stddev']<4 or p['foreground_entropy']<3 or len(points)<12
    return p


def compare_detail(a,b,*,controls_passed=True):
    # The visual-review flag is never fabricated: automatic failure is unresolved.
    result=verify_photos(a,b,controls_passed=controls_passed,policy=POLICY_V3)
    result['policy_version']=POLICY_V3['version']
    return result


def score_pairs(coarse,pairs):
    intensity=np.stack([p['intensity'] for p in coarse]);gradient=np.stack([p['gradient'] for p in coarse])
    scores=[]
    for a,b in pairs:
        scores.append((float(np.clip(np.dot(intensity[a],intensity[b]),-1,1)),float(np.clip(np.dot(gradient[a],gradient[b]),-1,1))))
    return scores


def select_detail_pairs(pairs,scores,mandatory,*,maximum=1536):
    if len(pairs)!=len(scores) or maximum<len(set(mandatory)):raise ValueError('Invalid bounded review budget')
    chosen=list(dict.fromkeys(mandatory))
    if any(i<0 or i>=len(pairs) for i in chosen):raise ValueError('Invalid mandatory edge')
    seen=set(chosen)
    ranks=[sorted(range(len(pairs)),key=lambda i:(-scores[i][j],pairs[i])) for j in (0,1)]
    for ranked in zip(*ranks):
        for i in ranked:
            if len(chosen)>=maximum:return chosen
            if i not in seen:chosen.append(i);seen.add(i)
    return chosen


def material_uncertainty(pair,score,evidence=None,*,prior_possible_copy=False):
    if pair[0]==pair[1]:return False
    if evidence and evidence.get('link'):return False
    if prior_possible_copy:return True
    if score[0]>=POLICY_V3['material_coarse_correlation']:return True
    if evidence:
        aligned=evidence.get('aligned_foreground_ssim')
        if aligned is not None and aligned>=POLICY_V3['material_aligned_ssim']:return True
        if evidence.get('inliers',0)>=12 and evidence.get('inlier_fraction',0)>=.5:return True
    return False
