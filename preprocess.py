# Input path for dicom images of patients - every pateint has one folder by name of patient Id
# e.g. /home/alokdwivedi/dev/avanid/data/qarc/dicom

# Input path where completed patients folder to be moved
# e.g. /home/alokdwivedi/dev/avanid/data/qarc/completed_dicom

# Goal is to create one npy file for each patient (64x64x64) to the output folder
# e.g. output path - /home/alokdwivedi/dev/avanid/data/qarc/numpy

# enumerate all patient folders
# patient is top level folder name - check its a valid 6 digit number
# This will be patientDICOM_images_path e.g. /home/alokdwivedi/dev/avanid/data/qarc/dicom/201489
# get the path of RS file -> RS.<patient_id>.dcm (e.g. String.format("RS.%d.dcm, patient_id))
#  get image by calling
# preprocessDICOM.preprocess(patientDICOM_images_path,RTstrPath,zero=False)
# save npy file for patient to output dir ie to /home/alokdwivedi/dev/avanid/data/qarc/numpy

# move patient folder to 'completed' folder ie to /home/alokdwivedi/dev/avanid/data/qarc/completed_dicom
# if error during processing patient folder (e.g. no RS fiel or more than one RS file) then move to 'error' folder
# under /home/alokdwivedi/dev/avanid/data/qarc/completed_dicom

#PLAN
#step 1: get the input path for the dicom file - will use os.walk (more info later)
# store this in variable: e.g. dicom_images_path = "/home/alokdwivedi/dev/avanid/data/qarc/dicom"

#step 2: get path list for patients
#reuse code from original

#CODE
#pathlist = []
#RTstrfiles=[]

#for x in os.walk(path2dcm):
#    pathlist.append(x)
# pathlist.pop[0] -get rid of first argument returned from os.walk which is the root (dicom folder), we only want the paths for the patient folders

#step 3: check for valid patient ID
#for error checking use a boolean
#to go through each patient will need a for loop to get each path for each patient

# for i in pathlist:
#   temp_patient_path = i

# to get the ID it is the last part of the path
#get this using os.path.basename

#CODE:
# temp_ID = os.path.basename(temp_patient_path)

#then check that it is a 6 digit ID (validation)
#if temp_ID.length() != 6:
#   validID == false


#step 4: check is RS file exists for each patient
# in original we iterated through the whole file and stored in an array then got the 0th position (only need the first one)
# instead we want to see if the file exists - use boolean?
# if true continue with preprocessing
# if false move to error file


#inside for loop check for RS file
#RS_exists = os.path.exists(temp_patient_path)
#   if RS_exists == false or ValidID == false:
#       CODE FOR MOVING INTO ERROR - step 5
#   else:
#       CODE FOR CONTINUING TO PREPROCESSING


#step 5: dealing with error
# get destination path: /home/alokdwivedi/dev/avanid/data/qarc/completed_dicom/error, store in variable
# error_path = /home/alokdwivedi/dev/avanid/data/qarc/completed_dicom
#use shutil.move

#CODE:
#shutil.move(temp_patient_path, error_path)
#NB need to import shutil


#step 6: continue with preprocessing
#CODE:
#outputdir = /home/alokdwivedi/dev/avanid/data/qarc/completed_dicom
#fname = os.path.join(outputdir,patient_id + '.npy')
#print(f'Going to generate cropped {fname}')
#img = preprocessDICOM.preprocess(DICOMpath,RTstrPath,zero=False)

# NOT SURE - think the above line of code does the preprocessing of image i.e., cropping etc

#CODE:
# np.save(fname,img)

#next we move to the completed folder like we did with the error files
#shutil.move(temp_patient_path, output_dir)



import os
import string

from utilities import preprocessDICOM
import numpy as np
import shutil

# NOTE: Its assumed that RT Struct file will follow a naming convention of RS.<patient_id>.dcm
rt_struct_file_format = "RS.{:s}.dcm"


def valid_patient_id(patient_id):
# TODO: check if it's a valid integer
# Question: Can we assume that the patient ID will always be a 6 digit integer value?
    return True


def move_to_error_folder(patient_id_path, completed_dicom_images_path):
# TODO: move input path into the error folder using shutil
    pass


def move_to_completed_folder(patient_id_path, completed_dicom_images_path):
    # TODO: move the patient_id_path to the completed folder
    pass


def preprocess_all_dicom_images(dicom_images_path, completed_dicom_images_path, output_nympy_path):
    patient_ids = get_patient_ids(dicom_images_path)

    for patient_id in patient_ids:
        patient_id_path = os.path.join(dicom_images_path, patient_id)
        if valid_patient_id(patient_id):
            # NOTE: Its assumed that RT Struct file will follow a naming convention of RS.<patient_id>.dcm
            patient_rt_struct_file_name = rt_struct_file_format.format(patient_id)
            patient_rt_struct_file_path = os.path.join(patient_id_path,patient_rt_struct_file_name)
            # check if RS file exists in that patient
            if not os.path.exists(patient_rt_struct_file_path):
                print(f"Ignoring Patient Id {patient_id} as it does not have an RS file {patient_rt_struct_file_path}");
                # move this folder to error folder
                move_to_error_folder(patient_id_path, completed_dicom_images_path)
            patient_numpy_file_path = os.path.join(output_nympy_path, patient_id + ".npy")
            print(f'Going to generate cropped {patient_numpy_file_path}')
            img = preprocessDICOM.preprocess(patient_id_path, patient_rt_struct_file_path, zero=False)
            np.save(patient_numpy_file_path, img)
            print(f'{patient_numpy_file_path} Cropped Successfully')
            move_to_completed_folder(patient_id_path, completed_dicom_images_path)
        else:
            print(f"Ignoring Patient Id {patient_id} as its not a valid 6 digit integer");
            # move this folder to error folder
            move_to_error_folder(patient_id_path, completed_dicom_images_path)


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

dicom_images_path = "/home/alokdwivedi/dev/avanid/data/qarc/dicom"
completed_dicom_images_path = " /home/alokdwivedi/dev/avanid/data/qarc/completed_dicom"
output_nympy_path = " /home/alokdwivedi/dev/avanid/data/qarc/numpy"

preprocess_all_dicom_images(dicom_images_path,completed_dicom_images_path,output_nympy_path)






