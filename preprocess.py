# Input path for dicom images of patients - every pateint has one folder by name of patient Id
# e.g. /home/alokdwivedi/dev/avanid/data/qarc/dicom

# Input path where completed patients folder to be moved
# e.g. /home/alokdwivedi/dev/avanid/data/qarc/completed_dicom

# output path where numpy files for each patient will be stored
# e.g. - /home/alokdwivedi/dev/avanid/data/qarc/numpy


import os
from utilities import preprocessDICOM
import numpy as np
import shutil

# NOTE: Its assumed that RT Struct file will follow a naming convention of RS.<patient_id>.dcm
RT_STRUCT_FILE_FORMAT = "RS.{:s}.dcm"


def valid_patient_id(patient_id):
    # Question: Can we assume that the patient ID will always be a 6 digit integer value?
    valid = True
    if len(patient_id) != 6:
        valid = False
    else:
        try:
            patient_id = int(patient_id)
        except ValueError:
            # handle the exception
            valid = False
    return valid


# method has been checked with incorrect type and incorrect length in jupyter notebook
# Question: is try except the same as try catch but python syntax rather than java syntax


def move_to_error_folder(patient_id_path, completed_dicom_images_path, patient_id):
    # first get error folder path
    error_path = os.path.join(completed_dicom_images_path, "error")
    error_patient_id_path = os.path.join(error_path, patient_id)
    if os.path.exists(error_patient_id_path):
        shutil.rmtree(error_patient_id_path, ignore_errors=True)
    shutil.move(patient_id_path, error_path)


def move_to_completed_folder(patient_id_path, completed_dicom_images_path, patient_id):
    completed_patient_id_path = os.path.join(completed_dicom_images_path, patient_id)
    if os.path.exists(completed_patient_id_path):
        shutil.rmtree(completed_patient_id_path, ignore_errors=True)
    shutil.move(patient_id_path, completed_dicom_images_path)


def preprocess_all_dicom_images(dicom_images_path, completed_dicom_images_path, output_nympy_path):
    patient_ids = get_patient_ids(dicom_images_path)

    for patient_id in patient_ids:
        try:
            preprocess_one_patient(completed_dicom_images_path, dicom_images_path, output_nympy_path, patient_id)
        except Exception as e:
            print(f"An error for patient_id {patient_id}.\n{e}")
            # move this folder to error folder
            patient_id_path = os.path.join(dicom_images_path, patient_id)
            move_to_error_folder(patient_id_path, completed_dicom_images_path, patient_id)


def preprocess_one_patient(completed_dicom_images_path, dicom_images_path, output_nympy_path, patient_id):
    patient_id_path = os.path.join(dicom_images_path, patient_id)
    if valid_patient_id(patient_id):
        # NOTE: Its assumed that RT Struct file will follow a naming convention of RS.<patient_id>.dcm
        patient_rt_struct_file_name = RT_STRUCT_FILE_FORMAT.format(patient_id)
        patient_rt_struct_file_path = os.path.join(patient_id_path, patient_rt_struct_file_name)
        # check if RS file exists in that patient
        if os.path.exists(patient_rt_struct_file_path):
            patient_numpy_file_path = os.path.join(output_nympy_path, patient_id + ".npy")
            print(f'Going to generate cropped {patient_numpy_file_path}')
            img = preprocessDICOM.preprocess(patient_id_path, patient_rt_struct_file_path, zero=False)
            # img = [0, 1, 2]
            np.save(patient_numpy_file_path, img)
            print(f'{patient_numpy_file_path} Cropped Successfully')
            move_to_completed_folder(patient_id_path, completed_dicom_images_path, patient_id)
        else:
            print(f"Ignoring Patient Id {patient_id} as it does not have an RS file {patient_rt_struct_file_path}");
            # move this folder to error folder
            move_to_error_folder(patient_id_path, completed_dicom_images_path, patient_id)
    else:
        print(f"Ignoring Patient Id {patient_id} as its not a valid 6 digit integer");
        # move this folder to error folder
        move_to_error_folder(patient_id_path, completed_dicom_images_path, patient_id)


def get_patient_ids(dicom_images_path):
    patient_ids = []
    dir_walk_level = 0
    for (_, dirs, _) in os.walk(dicom_images_path):
        dir_walk_level += 1
        if (dir_walk_level == 1):
            patient_ids = dirs
        else:
            break
    return patient_ids


# Test this method
# TODO: These inputs will be parameterised and become part of overall workflow that involves
# downloading DICOM images from DICOM server and then preprocessing and moving to output folder for ML Models
dicom_images_path = "/home/alokdwivedi/dev/avanid/data/qarc/dicom"
completed_dicom_images_path = "/home/alokdwivedi/dev/avanid/data/qarc/completed_dicom"
output_nympy_path = "/home/alokdwivedi/dev/avanid/data/qarc/numpy"

preprocess_all_dicom_images(dicom_images_path, completed_dicom_images_path, output_nympy_path)
