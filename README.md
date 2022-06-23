# QARCutilities
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
Python script `preprocess.py` contains logic to preprocess all patient images. For every patient, once the numpy file has been successfully created, the original patient folder is moved to a specified completed folder. If any error is encountered during preprocessing than that patient folder is moved to an error folder under the specified completed folder


