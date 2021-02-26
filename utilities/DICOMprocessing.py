from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.convert.filenameconverter import FilenameConverter
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.output.niioutputadapter import NiiOutputAdapter
from dcmrtstruct2nii.exceptions import PathDoesNotExistException, ContourOutOfBoundsException
import logging
import os.path
import re
import numpy as np
from numpy import load, save
import nibabel as nib
from nilearn.image import resample_img
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import os
from os import path
from IPython.display import Markdown, display
import shutil
import mmap
import pandas as pd
import tkinter
from tkinter import filedialog

def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))

def list_rt_structs(rtstruct_file):
    """
    Lists the structures in an DICOM RT Struct file by name.
    :param rtstruct_file: Path to the rtstruct file
    :return: A list of names, if any structures are found
    """
    if not os.path.exists(rtstruct_file):
        raise PathDoesNotExistException(f'rtstruct path does not exist: {rtstruct_file}')

    rtreader = RtStructInputAdapter()
    rtstructs = rtreader.ingest(rtstruct_file, True)
    return [struct['name'] for struct in rtstructs]

def dcmrtstruct2nii(rtstruct_file, dicom_file, outputdir, structures=None, gzip=True, mask_background_value=0, mask_foreground_value=255, convert_original_dicom=True):
    """
    Converts A DICOM and DICOM RT Struct file to nii
    :param rtstruct_file: Path to the rtstruct file
    :param dicom_file: Path to the dicom file
    :param outputdir: Output path where the masks are written to
    :param structures: Optional, list of structures to convert
    :param gzip: Optional, output .nii.gz if set to True, default: True
    :raise InvalidFileFormatException: Raised when an invalid file format is given.
    :raise PathDoesNotExistException: Raised when the given path does not exist.
    :raise UnsupportedTypeException: Raised when conversion is not supported.
    :raise ValueError: Raised when mask_background_value or mask_foreground_value is invalid.
    """
    outputdir = os.path.join(outputdir, '')  # make sure trailing slash is there

    if not os.path.exists(rtstruct_file):
        raise PathDoesNotExistException(f'rtstruct path does not exist: {rtstruct_file}')

    if not os.path.exists(dicom_file):
        raise PathDoesNotExistException(f'DICOM path does not exists: {dicom_file}')

    if mask_background_value < 0 or mask_background_value > 255:
        raise ValueError(f'Invalid value for mask_background_value: {mask_background_value}, must be between 0 and 255')

    if mask_foreground_value < 0 or mask_foreground_value > 255:
        raise ValueError(f'Invalid value for mask_foreground_value: {mask_foreground_value}, must be between 0 and 255')

    if structures is None:
        structures = []

    os.makedirs(outputdir, exist_ok=True)

    filename_converter = FilenameConverter()
    rtreader = RtStructInputAdapter()

    rtstructs = rtreader.ingest(rtstruct_file)
    dicom_image = DcmInputAdapter().ingest(dicom_file)

    dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
    nii_output_adapter = NiiOutputAdapter()
    for rtstruct in rtstructs:
        if len(structures) == 0 or rtstruct['name'] in structures:
            logging.info('Working on mask {}'.format(rtstruct['name']))
            try:
                mask = dcm_patient_coords_to_mask.convert(rtstruct['sequence'], dicom_image, mask_background_value, mask_foreground_value)
            except ContourOutOfBoundsException:
                logging.warning(f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!')
                continue

            mask.CopyInformation(dicom_image)

            mask_filename = filename_converter.convert(f'mask_{rtstruct["name"]}')
            nii_output_adapter.write(mask, f'{outputdir}{mask_filename}', gzip)

    if convert_original_dicom:
        logging.info('Converting original DICOM to nii')
        nii_output_adapter.write(dicom_image, f'{outputdir}image', gzip)

    logging.info('Success!')

# --------------Dialog Window------------------------------------------------

#rtstruct_file = "/Users/chufal/Desktop/0617-292370/03.dcm"
#dicom_file = "/Users/chufal/Desktop/0617-292370/"
#outputdir = "/Users/chufal/Desktop/TestOutput/"
#rinputdir = "D:\\Test_QARC_ESO"
#outputdir = "D:\\Test_Output"

#rinputdir = "C:/Users/Admin/Google Drive/projects/pyradiomics/data/Export_validation_Dataset/validation2"
#outputdir = "C:/Users/Admin/Google Drive/projects/pyradiomics/data/Export_validation_Dataset/output2"

# Choose you images and mask directory

# root = tkinter.Tk()
# root.withdraw() #use to hide tkinter window
# currdir = os.getcwd()
# rinputdir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please select a directory with DICOM images')
# if len(rinputdir) > 0:
#     print (f"You choose '{rinputdir}' as your Main directory")

# # Choose your output folder for saving images and masks
# outputdir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Choose your output folder for saving NIFTI images and Masks')
# if len(outputdir) > 0:
#     print (f"You choose '{outputdir}' as your Images and Mask directory")


#---------Final code to convert batch of dicom files
def DICOM2NIFTI(inputdir,outputdir):
    errors = []
    dicom_folder_list = os.listdir(inputdir)
    if 'images' not in os.listdir(outputdir):
        os.mkdir(os.path.join(outputdir,"images"))
    if 'mask' not in os.listdir(outputdir):
        os.mkdir(os.path.join(outputdir,"masks"))
    for i in range(0,len(dicom_folder_list)):
        if os.path.isdir(os.path.join(inputdir,dicom_folder_list[i])):
            print("\n")
            printmd(f"**{dicom_folder_list[i]} is a directory**", color="purple")
            print("\n")
            dicom_file_path = os.path.join(inputdir,dicom_folder_list[i])
            r = re.compile("^RS.[0-9]*.dcm") # to search RT structure file. Change reg expression as per need 
            #r = re.compile("^RS.[A-Z]*-[0-9]*.dcm")
            dicom_list = os.listdir(dicom_file_path)
            files2examine = list(filter(r.findall, dicom_list))
            for file in files2examine:
                rtstruct_file = os.path.join(dicom_file_path,file)            
                try:
                    strlist = list_rt_structs(rtstruct_file)
                    print(f"\n{rtstruct_file}: This is  RT Structure file\n")
                    path2RTstr = rtstruct_file
                    try:
                        dcmrtstruct2nii(rtstruct_file,dicom_file_path,outputdir)
                        CTimageData = "image.nii.gz"               
                        #CTmaskData = "mask_gtv_2.nii.gz"
                        CTmaskData = "mask_gtv_1.nii.gz"
                        CTimageDataPath = os.path.join(outputdir,CTimageData) 
                        CTmaskDataPath = os.path.join(outputdir,CTmaskData)
                        NewImageFileName = dicom_folder_list[i] +"_" + CTimageData
                        NewMaskFileName = dicom_folder_list[i] +"_" + CTmaskData
                        os.rename(CTimageDataPath,os.path.join(outputdir,NewImageFileName))
                        os.rename(CTmaskDataPath,os.path.join(outputdir,NewMaskFileName))


                        shutil.move(os.path.join(outputdir,NewImageFileName),os.path.join(outputdir,"images"))
                        shutil.move(os.path.join(outputdir,NewMaskFileName),os.path.join(outputdir,"masks"))
                        # Make sure you have folder named 'images','masks and temp'
                    except:
                        print(f"\n\x1b[31m\"Folder: {dicom_file_path}: Error in DICOM Directory..... Trying other method\"\x1b[0m\n")
                        
                        r = re.compile("^RS.[0-9]*.dcm") # to search RT structure file. Change reg expression as per need
                        #r = re.compile("^RS.[A-Z]*-[0-9]*.dcm")
                        dicom_list = os.listdir(dicom_file_path)
                        newfiles2examine = list(filter(r.findall, dicom_list))
                        for num,newfile in enumerate(newfiles2examine):
                            if newfile != file:
                                newrtstruct_file = os.path.join(dicom_file_path,newfile)
                                print("\n")
                                printmd(f"**This File: {newrtstruct_file} --Not a valid RT Structure file....removing the file**", color = "green")
                                print("\n")
                                shutil.move(newrtstruct_file,os.path.join(outputdir,"temp"))
                                newtempname = dicom_folder_list[i] + "_" + newrtstruct_file[-13:]
                                os.rename(os.path.join(outputdir,"temp",newrtstruct_file[-13:]), os.path.join(outputdir,"temp",newtempname)) 
                        try:
                            dcmrtstruct2nii(rtstruct_file,dicom_file_path,outputdir)
                            CTimageData = "image.nii.gz"
                            CTmaskData = "mask_gtv_1.nii.gz"
                            CTimageDataPath = os.path.join(outputdir,CTimageData) 
                            CTmaskDataPath = os.path.join(outputdir,CTmaskData)
                            NewImageFileName = dicom_folder_list[i] +"_" + CTimageData
                            NewMaskFileName = dicom_folder_list[i] +"_" + CTmaskData
                            os.rename(CTimageDataPath,os.path.join(outputdir,NewImageFileName))
                            os.rename(CTmaskDataPath,os.path.join(outputdir,NewMaskFileName))
                            shutil.move(os.path.join(outputdir,NewImageFileName),os.path.join(outputdir,"images"))
                            shutil.move(os.path.join(outputdir,NewMaskFileName),os.path.join(outputdir,"masks"))
                            break
                        except:
                            print("\n")
                            printmd('***Can not resolve the error, quitting this case***', color="black")
                            print("\n")
                            errors.append(dicom_file_path)
                            break
                except:
                    printmd(f"**This File: {rtstruct_file} --Not a valid RT Structure file....removing the file**", color = "orange")
                    shutil.move(rtstruct_file,os.path.join(outputdir,"temp"))
                    tempname = dicom_folder_list[i][5:] + "_" + rtstruct_file[-6:]
                    os.rename(os.path.join(outputdir,"temp",rtstruct_file[-6:]), os.path.join(outputdir,"temp",tempname))
        else:
            print("\n")
            printmd(f"**{dicom_folder_list[i]} is not a directory**", color="blue")
            print("\n")

    if len(errors) == 0:
        print("\n")
        printmd("***All DICOM folders have NO ISSUES and thus could be converted to NIFTI image and mask***",color="black")
    else:
        print("\n")
        printmd("***These DICOM folders have issues and thus could not be converted:***",color="black")
        printmd("**This information is saved in a csv file named 'rtog_errors.csv' in current working directory**",color="black")
        df = pd.DataFrame(data={"FileName":errors})
        df.to_csv(os.path.join(outputdir,'rtog_errors.csv'), sep=',',index=False)
        errors    