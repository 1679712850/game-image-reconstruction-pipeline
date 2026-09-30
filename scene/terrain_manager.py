"""Optional local color completion, explicitly inferred and excluded from visible QA."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cv.mask import read_mask, save_mask


def complete_terrain(source_path, terrain, output_dir):
    with Image.open(source_path) as image: source=np.array(image.convert('RGBA'))
    rgb=source[:,:,:3]
    root=Path(output_dir)/'terrain'
    for layer in terrain:
        mask=read_mask(layer['visible_mask_path'])>0
        # Infer only bounded enclosed holes; exterior unknown ground is never guessed as this class.
        holes=(~mask).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(holes,8)
        fill=np.zeros_like(mask)
        h,w=mask.shape
        for index in range(1,count):
            x,y,bw,bh,area=stats[index]
            if x>0 and y>0 and x+bw<w and y+bh<h and area<=max(64,int(mask.sum()*.25)):
                fill |= labels==index
        complete=mask|fill
        layer['complete_mask_path']=save_mask(complete,root/f"{layer['category']}_complete_mask.png")
        layer['occluded_mask_path']=save_mask(fill,root/f"{layer['category']}_inferred_mask.png")
        layer['completion_status']='opencv_telea_inferred' if fill.any() else 'no_enclosed_holes'
        if fill.any():
            # Erase all non-terrain inputs so adjacent objects cannot supply inpainting color.
            distance,nearest=cv2.distanceTransformWithLabels((~mask).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
            colors=rgb[mask]
            seeded=rgb.copy(); seeded[~mask]=colors[np.clip(nearest[~mask]-1,0,len(colors)-1)]
            completed=cv2.inpaint(seeded,fill.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
            rgba=np.dstack([completed,np.where(mask,source[:,:,3],fill.astype(np.uint8)*255)])
            path=root/f"{layer['category']}_complete.png"; Image.fromarray(rgba).save(path)
            layer['complete_asset_path']=str(path.resolve())
            layer['inferred_pixel_count']=int(fill.sum())
        else:
            layer['complete_asset_path']=layer['asset_path']; layer['inferred_pixel_count']=0
    return terrain
