from glob import glob
from sklearn.utils import shuffle
import tifffile
import cv2
import numpy as np
from patchify import patchify
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from skimage import measure
from tqdm import tqdm
from pathlib import Path

from skimage.segmentation import find_boundaries

seeds = [11920,1234,1337,42]


w0 = 10
sigma = 5


def make_weight_map(masks):
    masks = (masks > 0).astype(int)
    weight_map = np.zeros_like(masks)
    for i, mask in enumerate(masks):
        dist_transform = ndimage.distance_transform_edt(mask) 
        dist_transform *= 255.0/dist_transform.max()
        weight_map[i] = dist_transform
    return np.sum(weight_map,axis=0)

for training in tqdm(['base_training','base_training_patched','top_80_training','top_80_training_patched']):
    for seed in (seeds):
        for phase in ['train','val','test']:
            Path(f"Low_Magnification/{training}/{seed}/{phase}/distance_maps_low_conf").mkdir(parents=True, exist_ok=True)
            masks = glob(f'Low_Magnification/{training}/{seed}/{phase}/labels_low_conf/*.tif')
            
            for mask_path in (masks):
                mask = tifffile.imread(mask_path)
                original_size = mask.shape
                factor = 600/original_size[1]
                mask = cv2.resize(mask,(int(original_size[1]*factor),int(original_size[0]*factor)),interpolation=cv2.INTER_NEAREST)
                mask_split = np.array([(mask==i)*1 for i in range(1,int(np.max(mask)+1))])
                mask_split = np.array([x for x in mask_split if np.max(x) > 0])
                if len(mask_split) == 0:
                    tifffile.imwrite(mask_path.replace('labels_low_conf','distance_maps_low_conf'),np.ones(original_size))
                    continue
                weight = make_weight_map(mask_split)
                weight = weight / 255.
                weight = cv2.resize(weight,(original_size[1],original_size[0]),interpolation=cv2.INTER_NEAREST)
                tifffile.imwrite(mask_path.replace('labels_low_conf','distance_maps_low_conf'),weight)
