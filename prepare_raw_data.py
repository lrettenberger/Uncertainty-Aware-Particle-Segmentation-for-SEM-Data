from glob import glob
import tifffile
import cv2
from pathlib import Path
from tqdm import tqdm

LOW_MAG_SAMPLES_DIR = "./Low_Magnification/Samples"
LOW_MAG_HIGH_CONF_LABELS_DIR = "./Low_Magnification/High_Confidence_Labels"
LOW_MAG_LOW_CONF_LABELS_DIR = "./Low_Magnification/Low_Confidence_Labels"

HIGH_MAG_SAMPLES_DIR = "./High_Magnification/Samples"
HIGH_MAG_HIGH_CONF_LABELS_DIR = "./High_Magnification/High_Confidence_Labels"
HIGH_MAG_LOW_CONF_LABELS_DIR = "./High_Magnification/Low_Confidence_Labels"

# Write documentation of source files to a file
source_txt = open('./data_sources.txt','a')
source_txt.write(f'Source_Path,Image_Index,Image_Shape\n')

# Create low magnification samples/labels dirs
Path(LOW_MAG_SAMPLES_DIR).mkdir(parents=True, exist_ok=True)
Path(LOW_MAG_HIGH_CONF_LABELS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOW_MAG_LOW_CONF_LABELS_DIR).mkdir(parents=True, exist_ok=True)

# LOW MAGNIFICATION

# Read and sort the samples
training_imgs = glob('./Low_Mag_Source_Files/Confident-Labels/Positive-Examples/Training-Img*.png')
training_imgs = sorted(training_imgs,key=lambda x : int(x.split('/')[-1].split('Training-Img')[1].split('.')[0]))
imgs = glob('./Low_Mag_Source_Files/Confident-Labels/Positive-Examples/Img*.png')
imgs = sorted(imgs,key=lambda x : int(x.split('/')[-1].split('Img')[1].split('.')[0]))
round_3 = glob('./Low_Mag_Source_Files/Confident-Labels/Round-3/Img*.png')
round_3 = sorted(round_3,key=lambda x : int(x.split('/')[-1].split('Img')[1].split('.')[0]))
varied_brightness = glob('./Low_Mag_Source_Files/Confident-Labels/Varied-Brightness/Img*.tiff')
varied_brightness = sorted(varied_brightness,key=lambda x : int(x.split('/')[-1].split('.')[0].split('Img')[1]))

# All three data sources into one variable
all_samples = training_imgs + imgs + round_3 + varied_brightness

# Write all samples into respective dirs
train_img_counter = 0
for training_img_path in tqdm(all_samples):
    image = cv2.imread(training_img_path,-1)
    if '.png' in training_img_path:
        label_high_conf = tifffile.imread(training_img_path.replace('.png','_label.tif'))
    elif '.tiff' in training_img_path:
        label_high_conf = tifffile.imread(training_img_path.replace('.tiff','_label.tif'))
    label_low_conf = tifffile.imread(training_img_path.replace('.png','_label.tif').replace('Confident-Labels','Less-Certain_Labels'))
    tifffile.imwrite(f'{LOW_MAG_SAMPLES_DIR}/{train_img_counter}.tif',image)
    tifffile.imwrite(f'{LOW_MAG_LOW_CONF_LABELS_DIR}/{train_img_counter}.tif',label_low_conf)
    tifffile.imwrite(f'{LOW_MAG_HIGH_CONF_LABELS_DIR}/{train_img_counter}.tif',label_high_conf)
    source_txt.write(f'{training_img_path},{LOW_MAG_SAMPLES_DIR}/{train_img_counter}.tif,{image.shape}\n')
    train_img_counter+=1
source_txt.flush()


# HIGH MAGNIFICATION

# Create high magnification samples/labels dirs
Path(HIGH_MAG_SAMPLES_DIR).mkdir(parents=True, exist_ok=True)
Path(HIGH_MAG_HIGH_CONF_LABELS_DIR).mkdir(parents=True, exist_ok=True)
Path(HIGH_MAG_LOW_CONF_LABELS_DIR).mkdir(parents=True, exist_ok=True)


# Read and sort the samples
all_samples = glob('./High_Mag_Source_Files/Less-Certain_Labels/*.tiff')

# Write all samples into respective dirs
train_img_counter = 0
for training_img_path in tqdm(all_samples):
    image = tifffile.imread(training_img_path)
    # The "Confident-Labels" and "Less-Certain_Labels" are switched by name for the high mag files. So its exactly the opposite of how one would expect it
    label_high_conf = tifffile.imread(training_img_path.replace('.tiff','_label.tif'))
    label_low_conf = tifffile.imread(training_img_path.replace('.tiff','_label.tif').replace('Less-Certain_Labels','Confident-Labels'))
    tifffile.imwrite(f'{HIGH_MAG_SAMPLES_DIR}/{train_img_counter}.tif',image)
    tifffile.imwrite(f'{HIGH_MAG_LOW_CONF_LABELS_DIR}/{train_img_counter}.tif',label_low_conf)
    tifffile.imwrite(f'{HIGH_MAG_HIGH_CONF_LABELS_DIR}/{train_img_counter}.tif',label_high_conf)
    source_txt.write(f'{training_img_path},{HIGH_MAG_SAMPLES_DIR}/{train_img_counter}.tif,{image.shape}\n')
    train_img_counter+=1
source_txt.flush()
 