from glob import glob
from sklearn.utils import shuffle
import tifffile
import numpy as np
from patchify import patchify
from skimage.segmentation import watershed
from skimage import measure
from tqdm import tqdm
from pathlib import Path
import random
import shutil
import numpy as np

sample_size = 1200
step = 200
seeds = [11920,1234,1337,42]

# Test split provided by seed 2023.
# To have the same test samples for all seeds to ensure same results
test_sample_indices_low_mag = [0,2,4,7,16,18,33,35]
test_sample_indices_high_mag = [0,2,4,7,18,33,26]

def clean_label(img):
    global_max_instance = 0
    masks_binary = np.zeros(img.shape)
    for i in range(1,np.max(img)+1):
        seeds   = measure.label((img==i)*1, background=0)
        masks   = (img==i)*1
        gt_mask = watershed(image=-((img==i)*1), markers=seeds, mask=masks, watershed_line=False)
        gt_mask += global_max_instance
        gt_mask[gt_mask==global_max_instance] = 0
        masks_binary += gt_mask
        global_max_instance = int(np.max(masks_binary))
    label = masks_binary.astype(np.uint8)
    return label

TRAIN_VAL_TEST_SPLIT = 0.8

# COMBINED

for seed in tqdm(seeds):

    random.seed(seed)
    np.random.seed(seed)

    Path(f"Combined/base_training/{seed}/train/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Combined/base_training/{seed}/train/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Combined/base_training/{seed}/train/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Combined/base_training/{seed}/test/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Combined/base_training/{seed}/test/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Combined/base_training/{seed}/test/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Combined/base_training/{seed}/val/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Combined/base_training/{seed}/val/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Combined/base_training/{seed}/val/labels_high_conf").mkdir(parents=True, exist_ok=True)

    low_mag_samples = shuffle(glob('Low_Magnification/Samples/*.tif'))
    val_split = round(len(low_mag_samples)*TRAIN_VAL_TEST_SPLIT)
    counter=0
    for sample in (low_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices_low_mag:
            # if the sample is of the pre calculated test split
            dir = 'test'
        shutil.copy(sample,f'Combined/base_training/{seed}/{dir}/samples/')
        high_conf_label = clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels')))
        tifffile.imwrite(f'Combined/base_training/{seed}/{dir}/labels_high_conf/{sample.split("/")[-1]}',high_conf_label)
        counter+=1
    
    counter_low_mag = counter
    counter_low_mag+=1
    high_mag_samples = shuffle(glob('High_Magnification/Samples/*.tif'))
    val_split = round(len(high_mag_samples)*TRAIN_VAL_TEST_SPLIT)
    counter=0
    for sample in (high_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices_high_mag:
            # if the sample is of the pre calculated test split
            dir = 'test'
        shutil.copy(sample,f'Combined/base_training/{seed}/{dir}/samples/{counter+counter_low_mag}.tif')
        high_conf_label = clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels')))
        tifffile.imwrite(f'Combined/base_training/{seed}/{dir}/labels_high_conf/{counter+counter_low_mag}.tif',high_conf_label)
        counter+=1
        
        
# HIGH MAG AS NEGATIVE FOR LOW MAG

for seed in tqdm(seeds):

    random.seed(seed)
    np.random.seed(seed)

    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/train/samples").mkdir(parents=True, exist_ok=True)
    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/train/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/train/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/test/samples").mkdir(parents=True, exist_ok=True)
    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/test/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/test/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/val/samples").mkdir(parents=True, exist_ok=True)
    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/val/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/val/labels_high_conf").mkdir(parents=True, exist_ok=True)

    low_mag_samples = shuffle(glob('Low_Magnification/Samples/*.tif'))
    val_split = round(len(low_mag_samples)*TRAIN_VAL_TEST_SPLIT)
    counter=0
    for sample in (low_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices_low_mag:
            # if the sample is of the pre calculated test split
            dir = 'test'
        shutil.copy(sample,f'High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/{dir}/samples/')
        high_conf_label = clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels')))
        tifffile.imwrite(f'High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/{dir}/labels_high_conf/{sample.split("/")[-1]}',high_conf_label)
        counter+=1
    
    counter_low_mag = counter
    counter_low_mag+=1
    high_mag_samples = shuffle(glob('High_Magnification/Samples/*.tif'))
    val_split = round(len(high_mag_samples)*TRAIN_VAL_TEST_SPLIT)
    counter=0
    for sample in (high_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices_high_mag:
            # if the sample is of the pre calculated test split
            dir = 'test'
        shutil.copy(sample,f'High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/{dir}/samples/{counter+counter_low_mag}.tif')
        high_conf_label = np.zeros_like(clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels'))))
        tifffile.imwrite(f'High_Mag_As_Negative_For_Low_Mag/base_training/{seed}/{dir}/labels_high_conf/{counter+counter_low_mag}.tif',high_conf_label)
        counter+=1
        

# LOW MAG AS NEGATIVE FOR HIGH MAG

for seed in tqdm(seeds):

    random.seed(seed)
    np.random.seed(seed)

    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/train/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/train/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/train/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/test/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/test/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/test/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/val/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/val/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/val/labels_high_conf").mkdir(parents=True, exist_ok=True)

    low_mag_samples = shuffle(glob('Low_Magnification/Samples/*.tif'))
    val_split = round(len(low_mag_samples)*TRAIN_VAL_TEST_SPLIT)
    counter=0
    for sample in (low_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices_low_mag:
            # if the sample is of the pre calculated test split
            dir = 'test'
        shutil.copy(sample,f'Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/{dir}/samples/')
        high_conf_label = np.zeros_like(clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels'))))
        tifffile.imwrite(f'Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/{dir}/labels_high_conf/{sample.split("/")[-1]}',high_conf_label)
        counter+=1
    
    counter_low_mag = counter
    counter_low_mag+=1
    high_mag_samples = shuffle(glob('High_Magnification/Samples/*.tif'))
    val_split = round(len(high_mag_samples)*TRAIN_VAL_TEST_SPLIT)
    counter=0
    for sample in (high_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices_high_mag:
            # if the sample is of the pre calculated test split
            dir = 'test'
        shutil.copy(sample,f'Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/{dir}/samples/{counter+counter_low_mag}.tif')
        high_conf_label = clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels')))
        tifffile.imwrite(f'Low_Mag_As_Negative_For_High_Mag/base_training/{seed}/{dir}/labels_high_conf/{counter+counter_low_mag}.tif',high_conf_label)
        counter+=1