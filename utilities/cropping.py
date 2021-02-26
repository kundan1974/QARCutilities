def CropDicom(Path2ImgMsk="./Esophagus_NIFTI/",
            outpath="./Esophagus_224x224x224/",
            resample=1,
            MIN_BOUND = -1000.0,
            MAX_BOUND = 500.0,
            cropx=64,
            cropy=64,
            cropz=64,
            zero_centering=True,
            fill_value=0):


    import os
    import nibabel as nib
    from nilearn.image import resample_img
    import numpy as np
    from numpy import load, save
    import pandas as pd

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

    resamp = np.eye(3)
    resamp = np.where(resamp != 1,resamp,resample)

    # Loading nifti image and labels
    #os.remove(os.path.join(output_path,".DS_Store"))
    Path2Img = os.path.join(Path2ImgMsk,'images')
    Path2Msk = os.path.join(Path2ImgMsk,'masks')
    ImgNames = os.listdir(Path2Img)
    MskNames = os.listdir(Path2Msk)
    ImgNames.sort()
    MskNames.sort()


    cropped_img_list = []
    ID_list = []
    for i in range(0,len(ImgNames)):
        logger.info(f'Going to start cropping for {ImgNames[i]} images')
        if len(ImgNames) == len(MskNames):
            if ImgNames[i].split('_')[0] == MskNames[i].split('_')[0]:
                ID = ImgNames[i].split('_')[0]
                image_full_path = os.path.join(Path2Img,ImgNames[i])
                # Use NIB library to load zipped NII images
                # https://github.com/nipy/nibabel/blob/30c0bc561e0f34763c415a3fdc2b39cf0789a4ea/doc/source/images_and_memory.rst
                nifti_img = nib.load(image_full_path) # Loading nifti image
                nifti_img = resample_img(nifti_img, 
                                         target_affine=resamp, # resampling(1,1,1)
                                         interpolation='linear',
                                         fill_value=fill_value) 
                nifti_img_array = np.array(nifti_img.dataobj) # loaded image is converted to numpy array
                #nifti_img_array = np.rot90(nifti_img_array, k=3, axes=(0, 1)) # rotating image for proper orientation
                
                mask_full_path = os.path.join(Path2Msk,MskNames[i])
                nifti_msk = nib.load(mask_full_path) # Loading nifti mask
                nifti_msk = resample_img(nifti_msk, target_affine=resamp,
                                         interpolation='nearest',
                                         fill_value=fill_value) # resamling(1,1,1)
                nifti_msk_array = np.array(nifti_msk.dataobj) # loaded mask is converted to numpy array
                #nifti_msk_array = np.rot90(nifti_msk_array, k=3, axes=(0, 1)) # rotating mask for proper orientation
                
                result = np.where(nifti_msk_array == 255) # Get the index of elements with value 255
                z_min,z_max = result[2].min(),result[2].max()
                z_mid = (z_max-z_min)//2 
                z_midvalue = z_min + z_mid  # this is the position of the middle most slice along z axis
                
                centroide = []
                x_values = []
                y_values = []
                for value in range(z_min,z_max):
                    x_val = []
                    y_val = []
                    for j,val in enumerate(result[2]):
                        if val == value:
                            x_val.append(result[1][j])
                            y_val.append(result[0][j])
            
                    x_midvalues = np.array(x_val)
                    y_midvalues = np.array(y_val)
                    x_values.append(x_val)
                    y_values.append(y_val)

                    centroide.append([sum(x_midvalues)//len(x_midvalues),sum(y_midvalues)//len(y_midvalues)])
                # getting cropping parameters
                cent_array = np.array(centroide) # Converting to numpy array
                mean_cent = np.around(cent_array.mean(axis=0)).astype(int) # mean
                std_cent = np.around(cent_array.std(axis=0)).astype(int) # standard deviation
                x_meanmax = mean_cent[1] + (cropx//2)
                x_meanmin = mean_cent[1] - (cropx//2)
                y_meanmax = mean_cent[0] + (cropy//2)
                y_meanmin = mean_cent[0] - (cropy//2)
                if z_midvalue < cropz//4: # to check if no of slices with label is less than 24 if yes then set min value to zero
                    z_minval = 0
                else:
                    z_minval = z_midvalue - (cropz//2)
                z_maxval = z_midvalue + (cropz//2)
                print(x_meanmin,x_meanmax,y_meanmin,y_meanmax,z_minval,z_maxval)
                cropped_img = nifti_img_array[x_meanmin:x_meanmax,y_meanmin:y_meanmax,z_minval:z_maxval]
                print(cropped_img.min(),cropped_img.max())
                cropped_img = normalize(cropped_img) # image normalization (HU values converted between 0 and 1 after removing values >400)
                if zero_centering:
                    cropped_img = zero_center(cropped_img)
                cropped_img_list.append(cropped_img.shape)
                ID_list.append(ID)
                save(outpath + '/' + ID + '.npy',cropped_img)# Save cropped image as numpy array at selected loacation
                logger.info(f'Finished cropping for {ImgNames[i]} images')
            else:
                logger.error(f'Mask {MskNames[i]} does not matches with image {ImgNames[i]}')
        else:
            logger.error(f'Please check IMAGE folder and MASK folder. They have different length')
            
    df = pd.DataFrame(cropped_img_list,columns=['X','Y','Z'])
    df['CRnumber'] = ID_list
    save_csv_path = os.path.join(outpath,'converted_list.csv')
    df.to_csv(save_csv_path,columns=['CRnumber','X','Y','Z'])
    logger.info(f'{len(ImgNames)} images are cropped. Details are saved in a file named "converted_list.csv" and path to file is:{outpath}')
    print(f'{len(ImgNames)} images are cropped. Details are saved in a file named "converted_list.csv" and path to file is:{outpath}')
    return df