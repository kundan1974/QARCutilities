def DynamicCrop(Path2ImgMsk='../imagedata/workingFolder/',
                outpath='../imagedata/workingOutput/',
                resample=1,
                zero_centering = False,
                fill_value = 0,
                crop_fact = 32,
                crop_length=64):
    # Importing libraries
    import os
    import nibabel as nib
    from nilearn.image import resample_img
    import numpy as np
    from numpy import load, save
    import pandas as pd
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    
    import logging
    logger = logging.getLogger()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Setup file handler
    fhandler  = logging.FileHandler('my.log')
    fhandler.setLevel(logging.DEBUG)
    fhandler.setFormatter(formatter)

    # Configure stream handler for the cells
    chandler = logging.StreamHandler()
    chandler.setLevel(logging.DEBUG)
    chandler.setFormatter(formatter)

    # Add both handlers
    logger.addHandler(fhandler)
    logger.addHandler(chandler)
    logger.setLevel(logging.INFO)

    # Show the handlers
    logger.handlers

    # Creating functions to be used later
    def normalize(image):
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1
        image[image<0] = 0.
        return image

    # Zero Centering
    # As a final preprocessing step, it is advisory to zero center your data 
    # so that your mean value is 0. To do this you simply subtract the mean pixel value from all pixels
    def zero_center(image):
        image = image - image.mean()
        return image

    # creating array to resample image
    resamp = np.eye(3)
    resamp = np.where(resamp != 1,resamp,resample)


    # Loading nifti image and labels
    #os.remove(os.path.join(output_path,".DS_Store"))
    Path2Img = os.path.join(Path2ImgMsk,'images')
    Path2Msk = os.path.join(Path2ImgMsk,'masks')
    ImgNames = os.listdir(Path2Img)
    ImgNames.remove('.DS_Store')
    MskNames = os.listdir(Path2Msk)
    MskNames.remove('.DS_Store')
    ImgNames.sort()
    MskNames.sort()
    cropped_img_list = []
    ID_list = []
    
    # Creating cropped numpy image in batch
    cropped_img_list = []
    ID_list = []
    for m in range(0,len(ImgNames)):
        logger.info(f'Going to start cropping for {ImgNames[m]} images')
        if len(ImgNames) == len(MskNames):
            if ImgNames[m].split('_')[0] == MskNames[m].split('_')[0]:
                ID = ImgNames[m].split('_')[0]
                image_full_path = os.path.join(Path2Img,ImgNames[m])

                nifti_img = nib.load(image_full_path) # Loading nifti image
                nifti_img = resample_img(nifti_img, 
                                         target_affine=resamp, # resampling(1,1,1)
                                         interpolation='linear',
                                         fill_value=fill_value) 
                nifti_img_array = np.array(nifti_img.dataobj) # loaded image is converted to numpy array
                #nifti_img_array = np.rot90(nifti_img_array, k=3, axes=(0, 1)) # rotating image for proper orientation

                mask_full_path = os.path.join(Path2Msk,MskNames[m])
                nifti_msk = nib.load(mask_full_path) # Loading nifti mask
                nifti_msk = resample_img(nifti_msk, target_affine=resamp,
                                         interpolation='nearest',
                                         fill_value=fill_value) # resamling(1,1,1)
                nifti_msk_array = np.array(nifti_msk.dataobj) # loaded mask is converted to numpy array


                result = np.where(nifti_msk_array == 255) # Get the index of elements with value 255
                z_min,z_max = result[2].min(),result[2].max()
                z_mid = (z_max-z_min)//2 
                z_midvalue = z_min + z_mid  # this is the position of the middle most slice along z axis

                x_values = []
                y_values = []
                for value in range(z_min,z_max):
                    x_val = []
                    y_val = []
                    for j,val in enumerate(result[2]):
                        if val == value:
                            x_val.append(result[0][j])
                            y_val.append(result[1][j])

                    x_midvalues = np.array(x_val)
                    y_midvalues = np.array(y_val)
                    x_values.append(x_val)
                    y_values.append(y_val)


                x1=[]
                y1=[]
                z1=[]
                coords = []
                for value in range(z_midvalue-(crop_length//2),z_midvalue+(crop_length//2)):
                    x_val = []
                    y_val = []
                    z_val = []
                    for j,val in enumerate(result[2]):
                        if val == value:
                            x1 = result[0][j]
                            y1 = result[1][j]
                            z1 = result[2][j]
                            x_val.append(x1)
                            y_val.append(y1)
                            z_val.append(z1) 

                    x_val_arr = np.array(x_val)
                    y_val_arr = np.array(y_val)
                    z_val_arr = np.array(z_val)
                    coords.append([np.round(x_val_arr.mean()),np.round(y_val_arr.mean()),np.round(z_val_arr.mean())])


                nan_list = []
                z = len(coords)
                for i in range(z):
                    if np.isnan(coords[i]).sum() > 0:
                        nan_list.append(0)
                    else:
                        nan_list.append(1)
                nan_index = np.where(np.array(nan_list)==0) # index for nan valued cordinates in a list

                last_nan = []
                first_nan = []
                for j in nan_index[0]:
                    if j >= nan_list.index(1):
                        last_nan.append(j)
                    else:
                        first_nan.append(j)
                final_coords = []       
                check=0
                for i in range(len(nan_list)):
                    if nan_list[i] == 0:
                        check=check+1
                        if check <= nan_list.index(1):
                            final_coords.append(coords[nan_list.index(1)])
                        else:
                            final_coords.append(coords[last_nan[0]-1])
                    else:
                        final_coords.append(coords[i])


                crop_coords = []
                for coor in final_coords:
                    xmin = coor[0] - (crop_fact/2)
                    xmax = coor[0] + (crop_fact/2)
                    ymin = coor[1] - (crop_fact/2)
                    ymax = coor[1] + (crop_fact/2)
                    z = coor[2]
                    crop_coords.append([xmin,xmax,ymin,ymax,z])

                MIN_BOUND = nifti_img_array.min()
                MAX_BOUND = nifti_img_array.max()
                nifti_img_array = normalize(nifti_img_array)
                img_crop = np.zeros((crop_fact,crop_fact,len(final_coords)))

                for i in range(len(crop_coords)):
                    img_crop[0:,0:,i] = nifti_img_array[int(crop_coords[i][0]):int(crop_coords[i][1]),
                                                        int(crop_coords[i][2]):int(crop_coords[i][3]),
                                                        int(crop_coords[i][4])]
                    #MIN_BOUND = img_crop.min()
                    #MAX_BOUND = img_crop.max()
                    #img = normalize(img_crop)
                    if zero_centering:
                        img_crop = zero_center(img_crop)
                    img = img_crop
                cropped_img_list.append(img.shape)
                ID_list.append(ID)
                save(outpath + '/' + ID + '.npy',img)# Save cropped image as numpy array at selected loacation
                logger.info(f'Finished cropping for {ImgNames[m]} images')
            else:
                logger.error(f'Mask {MskNames[m]} does not matches with image {ImgNames[m]}')
        else:
            logger.error(f'Please check IMAGE folder and MASK folder. They have different length')
    df = pd.DataFrame(cropped_img_list,columns=['X','Y','Z'])
    df['CRnumber'] = ID_list
    save_csv_path = os.path.join(outpath,'converted_list.csv')
    df.to_csv(save_csv_path,columns=['CRnumber','X','Y','Z'])
    logger.info(f'{len(ImgNames)} images are cropped. Details are saved in a file named "converted_list.csv" and path to file is:{outpath}')
    return df