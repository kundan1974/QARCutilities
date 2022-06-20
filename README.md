# QARCutilities
utilities functions for QARC related project

## DICOM overview
Refer here https://dicom.nema.org/Dicom/Geninfo/brochure/RTAAPM.HTM

## Project Overview
All work done so far is available in this repo https://github.com/kundan1974/QARC-E001

Check out this for preprocessing overview
https://github.com/kundan1974/QARC-E001/blob/master/README.md#image-preprocessing--labels

## Misc Notes - Troubleshooting points
 - Need a requirements.txt to manage python dependencies. New module install needed
   - numpy
   - SimpleITK
   - scipy
   - scikit-image -> I had issue with scimage as I tried scimage instead of scikit-image as it was not obvious which one I needed
   - pydicom
   - matplotlib
 - There was circular dependency in ~/dev/github/QARCutilities/dcmrtstruct2nii/__init__.py 
    for now removed circular inclusion of dcmrtstruct2nii
 - I have added a superset of requirements.txt using pip command once I had a working setup
`pip freeze > requirements.txt`
 - Key python modules can now be installed by using script install_python_deps.sh
 - 
