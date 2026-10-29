"""Label-blind photographic evidence. dHash schedules review, never establishes identity."""
import hashlib
import math
import numpy as np

POLICY = {
    'version': 'photo-evidence-v2', 'size': 256, 'background_max': 8,
    'sift_features': 800, 'ratio_test': .75, 'ransac_pixels': 3.0,
    'minimum_inliers': 12, 'minimum_inlier_fraction': .85,
    'minimum_spatial_coverage': .15, 'minimum_overlap': .70,
    'minimum_aligned_ssim': .95, 'maximum_normalized_mae': .02,
    'minimum_gray_stddev': 4.0, 'minimum_entropy': 3.0,
    'minimum_features_for_false_positive_review': 40,
    'maximum_distinct_review_ssim': .80, 'verifier_rng_seed': 11,
    'scope': 'Technical same-photo copies under resize/JPEG/mild exposure/geometric transforms; not patient/eye identity or clinical adjudication',
    'threshold_basis': 'Fixed photographic transform/noise tolerances and synthetic controls; never class support, split sizes or model outcomes',
}


def prepare_photo(rgb, raw_sha256=None, policy=POLICY):
    import cv2
    from PIL import Image
    cv2.setNumThreads(1)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Require decoded EXIF-oriented RGB')
    pixel_sha = hashlib.sha256(str((rgb.shape[1],rgb.shape[0])).encode()+rgb.tobytes()).hexdigest()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > policy['background_max']).astype(np.uint8)
    # Largest external outline excludes shared black canvas; fill internal vessels.
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:cv2.drawContours(filled,[max(contours,key=cv2.contourArea)],-1,1,-1)
    ys,xs = np.nonzero(filled)
    size = policy['size']
    full = cv2.resize(gray,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    valid = len(xs)>0 and filled.mean()>=.20
    if valid:
        box=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
        x0,y0,x1,y1=box
        crop=cv2.resize(gray[y0:y1,x0:x1],(size,size),interpolation=cv2.INTER_AREA)
        fm=cv2.resize(filled[y0:y1,x0:x1],(size,size),interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        box=None;crop=np.zeros((size,size),np.uint8);fm=np.zeros_like(crop,dtype=bool)
    values=crop[fm]
    hist=np.bincount(values,minlength=256).astype(float)
    probs=hist[hist>0]/max(len(values),1)
    entropy=float(-(probs*np.log2(probs)).sum())
    std=float(values.std()) if len(values) else 0.
    # Fixed intensity scale avoids amplifying tiny JPEG/percentile changes.
    norm=crop.astype(np.float32)/255.0
    sift=cv2.SIFT_create(nfeatures=policy['sift_features'])
    keypoints,descriptors=sift.detectAndCompute(crop,(fm*255).astype(np.uint8))
    points=np.array([k.pt for k in keypoints],np.float32).reshape(-1,2)
    # Independent implementation: pack row-wise comparison bits, not runner helper.
    small=np.asarray(Image.fromarray(rgb).convert('L').resize((9,8),Image.Resampling.LANCZOS))
    dhash=int.from_bytes(np.packbits((small[:,:-1]>small[:,1:]).ravel()).tobytes(),'big')
    return {'raw_sha256':raw_sha256,'pixel_sha256':pixel_sha,'dhash':f'{dhash:016x}',
            'width':rgb.shape[1],'height':rgb.shape[0],'foreground_box':box,
            'foreground_fraction':float(filled.mean()),'foreground_stddev':std,
            'foreground_entropy':entropy,'features':len(points),
            'low_information':not valid or std<policy['minimum_gray_stddev'] or entropy<policy['minimum_entropy'] or len(points)<policy['minimum_inliers'],
            '_full':full,'_gray':norm,'_mask':fm,'_points':points,'_descriptors':descriptors}


def _ssim(a,b,mask):
    from skimage.metrics import structural_similarity
    if mask.sum()<64:return None
    _,image=structural_similarity(a,b,data_range=1.,full=True)
    return float(image[mask].mean())


def verify_photos(a,b,*,distinct_visual_review=False,controls_passed=True,policy=POLICY):
    import cv2
    cv2.setRNGSeed(policy['verifier_rng_seed'])
    mask=a['_mask']&b['_mask']
    result={'classification':'unresolved_candidate','link':False,
            'reason':'Independent copy evidence insufficient; verification failure alone is not a false positive',
            'full_frame_ssim':_ssim(a['_full'],b['_full'],np.ones_like(mask)),
            'foreground_ssim_unregistered':_ssim(a['_gray'],b['_gray'],mask),
            'low_information':a['low_information'] or b['low_information'],
            'mutual_matches':0,'inliers':0,'inlier_fraction':0.,'spatial_coverage_a':0.,'spatial_coverage_b':0.,
            'aligned_foreground_ssim':None,'aligned_normalized_mae':None,'overlap':0.,
            'clinical_adjudication':False,'controls_passed':controls_passed}
    if ((a['raw_sha256'] is not None and a['raw_sha256']==b['raw_sha256']) or a['pixel_sha256']==b['pixel_sha256']):
        result.update(classification='confirmed_duplicate',link=True,reason='Verified identical raw bytes or canonical decoded pixels')
        return result
    da,db=a['_descriptors'],b['_descriptors']
    if da is not None and db is not None and min(len(da),len(db))>=2:
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        def ratios(x,y):
            return {m.queryIdx:m.trainIdx for matches in matcher.knnMatch(x,y,k=2) if len(matches)==2 for m,n in [matches] if m.distance<policy['ratio_test']*n.distance}
        forward,reverse=ratios(da,db),ratios(db,da)
        mutual=[(i,j) for i,j in sorted(forward.items()) if reverse.get(j)==i]
        result['mutual_matches']=len(mutual)
        if len(mutual)>=4:
            pa=np.array([a['_points'][i] for i,j in mutual]);pb=np.array([b['_points'][j] for i,j in mutual])
            h,inliers=cv2.findHomography(pa,pb,cv2.RANSAC,policy['ransac_pixels'],maxIters=2000,confidence=.999)
            if h is not None and inliers is not None and np.isfinite(h).all():
                good=inliers.ravel().astype(bool);result['inliers']=int(good.sum());result['inlier_fraction']=float(good.mean())
                if good.sum()>=3:
                    result['spatial_coverage_a']=float(cv2.contourArea(cv2.convexHull(pa[good]))/max(a['_mask'].sum(),1))
                    result['spatial_coverage_b']=float(cv2.contourArea(cv2.convexHull(pb[good]))/max(b['_mask'].sum(),1))
                size=policy['size'];warped=cv2.warpPerspective(a['_gray'],h,(size,size))
                wm=cv2.warpPerspective(a['_mask'].astype(np.uint8),h,(size,size),flags=cv2.INTER_NEAREST)>0
                common=wm&b['_mask'];common=cv2.erode(common.astype(np.uint8),np.ones((7,7),np.uint8))>0
                result['overlap']=float(min(common.sum()/max(a['_mask'].sum(),1),common.sum()/max(b['_mask'].sum(),1)))
                result['aligned_foreground_ssim']=_ssim(warped,b['_gray'],common)
                if common.any():result['aligned_normalized_mae']=float(np.abs(warped-b['_gray'])[common].mean())
    probable=(controls_passed and not result['low_information'] and result['inliers']>=policy['minimum_inliers']
              and result['inlier_fraction']>=policy['minimum_inlier_fraction']
              and min(result['spatial_coverage_a'],result['spatial_coverage_b'])>=policy['minimum_spatial_coverage']
              and result['overlap']>=policy['minimum_overlap']
              and result['aligned_foreground_ssim'] is not None and result['aligned_foreground_ssim']>=policy['minimum_aligned_ssim']
              and result['aligned_normalized_mae']<=policy['maximum_normalized_mae'])
    if probable:
        result.update(classification='strongly_supported_probable_copy',link=True,reason='Spatially distributed mutual SIFT/RANSAC agreement plus aligned foreground photometric/structural agreement; not clinical identity')
    elif (controls_passed and distinct_visual_review and not result['low_information']
          and min(a['features'],b['features'])>=policy['minimum_features_for_false_positive_review']
          and result['foreground_ssim_unregistered'] is not None
          and result['foreground_ssim_unregistered']<=policy['maximum_distinct_review_ssim']
          and result['mutual_matches']<policy['minimum_inliers']):
        result.update(classification='evidence_supported_false_positive',reason='Hash match contradicted by reviewed different foreground structure and independent feature evidence under the declared copy-transform scope; no patient-identity conclusion')
    return result


def public_photo_record(photo):
    return {k:v for k,v in photo.items() if not k.startswith('_')}
