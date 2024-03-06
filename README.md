# Machine-Learning-Particle-Morphology-Estimation-Data


## Install

- Install [anaconda>=4.12.0](https://www.anaconda.com/download/)
- Create conda environment with `conda create --name <insert your env name> python=3.8`
- Activate conda environment with `conda activate <insert your env name>`
- Install dependencies `pip install -r requirements.txt`
  
## Usage
- Download `High_Mag_Source_Files.zip`  and `Low_Mag_Source_Files.zip` from [https://osf.io/f2y8w/](https://osf.io/f2y8w/)
- Unzip both zips in this directory
- Execute `prepare_raw_data.py`
- Execute `prepare_training_splits_high_magnification.py`
- Execute `prepare_training_splits_low_magnification.py`
- Execute `generate_distance_maps.py` if distance maps are required (e.g. for U-Net)
