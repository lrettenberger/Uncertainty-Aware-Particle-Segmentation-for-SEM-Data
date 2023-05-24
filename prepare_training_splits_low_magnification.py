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

# Skip 80% and Patching
ONLY_BASE_TRAINING = True

sample_size = 1200
step = 200
seeds = [11920,1234,1337,42]

# Test split provided by seed 2023.
# To have the same test samples for all seeds to ensure same results
test_sample_indices = [0,2,4,7,16,18,33,35]

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

# Base training.

for seed in tqdm(seeds):

    random.seed(seed)
    np.random.seed(seed)

    Path(f"Low_Magnification/base_training/{seed}/train/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training/{seed}/train/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training/{seed}/train/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Magnification/base_training/{seed}/test/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training/{seed}/test/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training/{seed}/test/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Magnification/base_training/{seed}/val/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training/{seed}/val/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training/{seed}/val/labels_high_conf").mkdir(parents=True, exist_ok=True)

    low_mag_samples = shuffle(glob('Low_Magnification/Samples/*.tif'))
    val_split = round(len(low_mag_samples)*TRAIN_VAL_TEST_SPLIT)
    counter=0
    for sample in (low_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices:
            # if the sample is of the pre calculated test split
            dir = 'test'
        shutil.copy(sample,f'Low_Magnification/base_training/{seed}/{dir}/samples/')
        low_conf_label = clean_label(tifffile.imread(sample.replace('Samples','Low_Confidence_Labels')))
        high_conf_label = clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels')))
        tifffile.imwrite(f'Low_Magnification/base_training/{seed}/{dir}/labels_low_conf/{sample.split("/")[-1]}',low_conf_label)
        tifffile.imwrite(f'Low_Magnification/base_training/{seed}/{dir}/labels_high_conf/{sample.split("/")[-1]}',high_conf_label)
        counter+=1

if ONLY_BASE_TRAINING:
    exit()


# only take top 80% particles (or remove 20% smallest particles from all segmentation masks)

for seed in tqdm(seeds):

    random.seed(seed)
    np.random.seed(seed)

    Path(f"Low_Magnification/top_80_training/{seed}/train/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training/{seed}/train/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training/{seed}/train/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Magnification/top_80_training/{seed}/test/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training/{seed}/test/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training/{seed}/test/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Magnification/top_80_training/{seed}/val/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training/{seed}/val/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training/{seed}/val/labels_high_conf").mkdir(parents=True, exist_ok=True)

    low_mag_samples = shuffle(glob('Low_Magnification/Samples/*.tif'))
    val_split = round(len(low_mag_samples)*TRAIN_VAL_TEST_SPLIT)


    particle_sizes = []
    for sample in (low_mag_samples):
        low_conf_label = clean_label(tifffile.imread(sample.replace('Samples','Low_Confidence_Labels')))
        particle_sizes.extend([np.sum(low_conf_label==i) for i in range(1,np.max(low_conf_label)+1)])
    particle_sizes = np.sort(particle_sizes)
    #'Removing lower 20%'
    size_threshold = particle_sizes[int(len(particle_sizes)*0.2)]

    counter=0
    for sample in (low_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices:
            # if the sample is of the pre calculated test split
            dir = 'test'
        shutil.copy(sample,f'Low_Magnification/top_80_training/{seed}/{dir}/samples/')
        low_conf_label = clean_label(tifffile.imread(sample.replace('Samples','Low_Confidence_Labels')))
        # remove small particles
        for i in range(1,np.max(low_conf_label)+1):
            particle_area = np.sum(low_conf_label==i)
            if particle_area < size_threshold:
                low_conf_label[low_conf_label==i] = 0
        low_conf_label = clean_label(low_conf_label)
        tifffile.imwrite(f'Low_Magnification/top_80_training/{seed}/{dir}/labels_low_conf/{sample.split("/")[-1]}',low_conf_label)
        high_conf_label = clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels')))
        # remove small particles
        for i in range(1,np.max(high_conf_label)+1):
            particle_area = np.sum(high_conf_label==i)
            if particle_area < size_threshold:
                high_conf_label[high_conf_label==i] = 0
        high_conf_label = clean_label(high_conf_label)
        tifffile.imwrite(f'Low_Magnification/top_80_training/{seed}/{dir}/labels_high_conf/{sample.split("/")[-1]}',high_conf_label)
        counter+=1
    
# Split into patches but keep all labels

for seed in tqdm(seeds):

    random.seed(seed)
    np.random.seed(seed)

    Path(f"Low_Magnification/base_training_patched/{seed}/train/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training_patched/{seed}/train/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training_patched/{seed}/train/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Magnification/base_training_patched/{seed}/test/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training_patched/{seed}/test/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training_patched/{seed}/test/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Magnification/base_training_patched/{seed}/val/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training_patched/{seed}/val/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/base_training_patched/{seed}/val/labels_high_conf").mkdir(parents=True, exist_ok=True)

    low_mag_samples = shuffle(glob('Low_Magnification/Samples/*.tif'))
    val_split = round(len(low_mag_samples)*TRAIN_VAL_TEST_SPLIT)
    counter=0
    for sample in (low_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices:
            # if the sample is of the pre calculated test split
            dir = 'test'
        image = tifffile.imread(sample)
        low_conf_label = clean_label(tifffile.imread(sample.replace('Samples','Low_Confidence_Labels')))
        high_conf_label = clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels')))
        image_patches = patchify(image, (sample_size, sample_size), step=step)
        label_patches_low_conf = patchify(low_conf_label, (sample_size, sample_size), step=step)
        label_patches_high_conf = patchify(high_conf_label, (sample_size, sample_size), step=step)
        patch_counter = 0
        for i in range(image_patches.shape[0]):
            for j in range(image_patches.shape[1]):
                image_patch = image_patches[i, j]
                label_patch_low_conf = clean_label(label_patches_low_conf[i, j])
                label_patch_high_conf = clean_label(label_patches_high_conf[i, j])
                tifffile.imwrite(f'Low_Magnification/base_training_patched/{seed}/{dir}/samples/{patch_counter}_{sample.split("/")[-1]}',image_patch)
                tifffile.imwrite(f'Low_Magnification/base_training_patched/{seed}/{dir}/labels_low_conf/{patch_counter}_{sample.split("/")[-1]}',label_patch_low_conf)
                tifffile.imwrite(f'Low_Magnification/base_training_patched/{seed}/{dir}/labels_high_conf/{patch_counter}_{sample.split("/")[-1]}',label_patch_high_conf)
                patch_counter+=1
        counter+=1

# split into patches 80%

for seed in tqdm(seeds):

    random.seed(seed)
    np.random.seed(seed)

    Path(f"Low_Magnification/top_80_training_patched/{seed}/train/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training_patched/{seed}/train/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training_patched/{seed}/train/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Magnification/top_80_training_patched/{seed}/test/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training_patched/{seed}/test/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training_patched/{seed}/test/labels_high_conf").mkdir(parents=True, exist_ok=True)

    Path(f"Low_Magnification/top_80_training_patched/{seed}/val/samples").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training_patched/{seed}/val/labels_low_conf").mkdir(parents=True, exist_ok=True)
    Path(f"Low_Magnification/top_80_training_patched/{seed}/val/labels_high_conf").mkdir(parents=True, exist_ok=True)

    low_mag_samples = shuffle(glob('Low_Magnification/Samples/*.tif'))
    val_split = round(len(low_mag_samples)*TRAIN_VAL_TEST_SPLIT)


    particle_sizes = []
    for sample in (low_mag_samples):
        low_conf_label = clean_label(tifffile.imread(sample.replace('Samples','Low_Confidence_Labels')))
        particle_sizes.extend([np.sum(low_conf_label==i) for i in range(1,np.max(low_conf_label)+1)])
    particle_sizes = np.sort(particle_sizes)
    #'Removing lower 20%'
    size_threshold = particle_sizes[int(len(particle_sizes)*0.2)]

    counter=0
    for sample in (low_mag_samples):
        if(tifffile.imread(sample).shape != (1200,1920)):
            raise Exception('Wrong image size! Aborting')
        dir = 'train'
        if counter >= val_split:
            dir = 'val'
        if int(sample.split('/')[-1].split('.')[0]) in test_sample_indices:
            # if the sample is of the pre calculated test split
            dir = 'test'
        image = tifffile.imread(sample)
        low_conf_label = clean_label(tifffile.imread(sample.replace('Samples','Low_Confidence_Labels')))
        # remove small particles
        for i in range(1,np.max(low_conf_label)+1):
            particle_area = np.sum(low_conf_label==i)
            if particle_area < size_threshold:
                low_conf_label[low_conf_label==i] = 0
        low_conf_label = clean_label(low_conf_label)
        
        hig_conf_label = clean_label(tifffile.imread(sample.replace('Samples','High_Confidence_Labels')))
        # remove small particles
        for i in range(1,np.max(hig_conf_label)+1):
            particle_area = np.sum(hig_conf_label==i)
            if particle_area < size_threshold:
                hig_conf_label[hig_conf_label==i] = 0
        hig_conf_label = clean_label(hig_conf_label)
        
        image_patches = patchify(image, (sample_size, sample_size), step=step)
        label_patches_low_conf = patchify(low_conf_label, (sample_size, sample_size), step=step)
        label_patches_high_conf = patchify(hig_conf_label, (sample_size, sample_size), step=step)
        patch_counter = 0
        for i in range(image_patches.shape[0]):
            for j in range(image_patches.shape[1]):
                image_patch = image_patches[i, j]
                label_patch_low_conf = clean_label(label_patches_low_conf[i, j])
                label_patch_high_conf = clean_label(label_patches_high_conf[i, j])
                tifffile.imwrite(f'Low_Magnification/top_80_training_patched/{seed}/{dir}/samples/{patch_counter}_{sample.split("/")[-1]}',image_patch)
                tifffile.imwrite(f'Low_Magnification/top_80_training_patched/{seed}/{dir}/labels_low_conf/{patch_counter}_{sample.split("/")[-1]}',label_patch_low_conf)
                tifffile.imwrite(f'Low_Magnification/top_80_training_patched/{seed}/{dir}/labels_high_conf/{patch_counter}_{sample.split("/")[-1]}',label_patch_high_conf)
                patch_counter+=1
        counter+=1