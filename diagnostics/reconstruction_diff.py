"""Alpha-aware SSIM, edge errors and color-coded reconstruction diagnostics."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from app.paths import read_rgba


def write_reconstruction_diff(source_path, reconstruction_path, output_path, lpips_enabled=False):
    source=np.asarray(read_rgba(source_path)).astype(np.float32)/255
    rebuilt=np.asarray(read_rgba(reconstruction_path)).astype(np.float32)/255
    if source.shape != rebuilt.shape: raise ValueError('Reconstruction dimensions differ')
    # Premultiply RGB so arbitrary RGB in transparent pixels cannot affect the metric.
    a=source[:,:,:3]*source[:,:,3:4]; b=rebuilt[:,:,:3]*rebuilt[:,:,3:4]
    sa,ba=source[:,:,3],rebuilt[:,:,3]
    missing=sa>ba+.03; extra=ba>sa+.03
    gray_a=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY); gray_b=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
    mu_a=cv2.GaussianBlur(gray_a,(11,11),1.5); mu_b=cv2.GaussianBlur(gray_b,(11,11),1.5)
    va=cv2.GaussianBlur(gray_a**2,(11,11),1.5)-mu_a**2
    vb=cv2.GaussianBlur(gray_b**2,(11,11),1.5)-mu_b**2
    cov=cv2.GaussianBlur(gray_a*gray_b,(11,11),1.5)-mu_a*mu_b
    ssim=((2*mu_a*mu_b+.01**2)*(2*cov+.03**2))/((mu_a**2+mu_b**2+.01**2)*(va+vb+.03**2))
    ea=cv2.Canny((gray_a*255).astype(np.uint8),60,150)>0
    eb=cv2.Canny((gray_b*255).astype(np.uint8),60,150)>0
    diff=np.max(np.abs(a-b),axis=2)
    heat=np.zeros((*sa.shape,3),np.uint8)
    heat[(diff>.12)| (ea^eb)]=(255,220,0); heat[missing]=(255,0,0); heat[extra]=(0,60,255)
    path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(heat).save(path)
    perceptual=None; status='disabled'
    if lpips_enabled:
        # No silent downloads. Opt-in requires lpips and locally cached pretrained weights.
        try:
            import torch, lpips
            weights=Path(torch.hub.get_dir())/'checkpoints'/'alexnet-owt-7be5be79.pth'
            if not weights.is_file():
                raise OSError('LPIPS backbone weights are not cached; automatic download disabled')
            model=lpips.LPIPS(net='alex',pretrained=True,verbose=False)
            with torch.inference_mode():
                aa=torch.from_numpy(a.transpose(2,0,1).copy())[None]*2-1
                bb=torch.from_numpy(b.transpose(2,0,1).copy())[None]*2-1
                perceptual=float(model(aa,bb).item())
            status='computed'
        except (ImportError,RuntimeError,OSError) as error:
            status=f'unavailable: {type(error).__name__}'
    return {'path':str(path.resolve()),'ssim':float(np.clip(ssim.mean(),-1,1)),
            'ssim_definition':'11x11 Gaussian luminance SSIM on premultiplied RGB; alpha separately measured',
            'lpips':perceptual,'lpips_status':status,'edge_difference':float((ea^eb).mean()),
            'alpha_gap':float(missing.mean()),'extra_alpha':float(extra.mean()),
            'mean_pixel_difference':float((3*np.abs(a-b).mean()+np.abs(sa-ba).mean())/4),
            'pixel_difference_definition':'Mean absolute difference of premultiplied RGB and alpha',
            'missing_pixel_count':int(missing.sum()),'extra_pixel_count':int(extra.sum()),
            'high_difference_pixel_count':int((diff>.12).sum())}
