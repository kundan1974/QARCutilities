#
# **** IMPORTANT ****
# Change these inout paths as per your environment before running bash script
# **** IMPORTANT ****
#

DICOM_IMAGES_PATH='/home/alokdwivedi/dev/avanid/data/qarc/dicom'
COMPLETED_DICOM_IMAGES_PATH='/home/alokdwivedi/dev/avanid/data/qarc/completed_dicom'
NUMPY_OUTPUT_PATH='/home/alokdwivedi/dev/avanid/data/qarc/numpy'

python3 preprocess.py ${DICOM_IMAGES_PATH} ${COMPLETED_DICOM_IMAGES_PATH} ${NUMPY_OUTPUT_PATH}