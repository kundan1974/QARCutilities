# QARC Utilities
utilities functions for QARC related project

## DICOM overview
Refer here https://dicom.nema.org/Dicom/Geninfo/brochure/RTAAPM.HTM

## Project Overview
All work done so far is available in this repo https://github.com/kundan1974/QARC-E001

Check out this for preprocessing overview
https://github.com/kundan1974/QARC-E001/blob/master/README.md#image-preprocessing--labels

## Python module dependencies
All python module dependency has been cpatured in requirements.txt. You can use `install_deps.sh` shell script. Alternatively run following command to get all Python modules installed in one go
`pip install -r requirements.txt`

## Preprocessing all patient images

- Python script `preprocess.py` contains logic to preprocess all patient images. For every patient, once the numpy file has been successfully created, the original patient folder is moved to a specified completed folder. If any error is encountered during preprocessing than that patient folder is moved to an error folder under the specified completed folder
In the overall workflow, DICOM images will be fetched from server and numpy files will be split in test and train sets for CNN model to use as inputs
- Shell script `preprocess.sh` runs python script `preprocess.py`. Here you can edit input/output folders to suite your environments
- `setup_preprocess.sh` bash scripts adds a crontab entry to schedule running `preprocess.sh` at 22:00 hours every day. This will ensure any new DICOM patient images added will be automatically processed on next run

## Misc 
https://github.com/thoraciclang/Deep_Lung - Models based on Cox Proportional Hazard Model and Kaplan Meir Analysis

Combining clinical data with images
https://ieeexplore.ieee.org/document/9661330/
https://youtu.be/u4H3KtGxM7I
