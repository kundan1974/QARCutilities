"""
This script contins various utility functions to be used when building 
and training Deep Learning models.

"""
def create_df(
                imgfolder='./Datafiles/Images/AllNumpyImages_211',
                datafile='path_to_data_file'
                ):
    
    """
    This function returns dataframe with IDs and their respective labels 
        for all patients included in QARC-Deeplearning study. 
    Parameters:
    -----------
    Image folder: str
        Path to folder having all the images to be included in the study. 
        These images are numpy format images of size (64,64,64) and are 
        ready to be used as input for CNN deep neural network.
    Datafile: sr
        Path to the master csv file having all relevant clinical information 
        for the all patients

    Returns: 
    --------
        pandas dataframe
                dataframe returned by this functuion contains IDs and labels 
                for only those patients whose numpy image data is present 
                in image folder
    
    """
    import pandas as pd
    import os
    img_files = os.listdir(imgfolder) # list of files present in Images folder
    if datafile[-4:] == 'xlsx':
        df = pd.read_excel(datafile)
    else:
        df = pd.read_csv(datafile) # main dataframe
    
    # Extracting ID from filename 
    # (These filename are of images present in the Images folder)
    IDs = []
    for i in range(len(img_files)):
        IDs.append(int(img_files[i][:-4]))

    # As per the file name ---- extracting the same patientID as 
    # per the file name from the main dataframe
    # Saving these extracted IDs matched with the image name in a list

    listID = []
    for i in df.ID:
        for j in range(len(IDs)):
            if i == IDs[j]:
                listID.append(i)

    # On the basis of this saved list we extract information for labels from the 
    # main dataset and saving it as a new dataframe
    columns = list(df)
    data = []
    df_new = pd.DataFrame(columns=columns)

    for i in range(len(listID)):
        for j in df.ID:
            if listID[i] == j:
                value = (df[df.ID==j].values).tolist()[0]
                zipped = zip(columns,value)
                IDdict = dict(zipped)
                data.append(IDdict)

    df_new = df_new.append(data,True)
    
    return df_new 

def split_traintest(
                    path2allLabels='./Datafiles/Labels/labels_211.csv',
                    s1=0.15,
                    s2=0.15,
                    make_dirs = False
                    ):
    """
    Parameters:
    -----------
        path2allLabels: str
            is a path to csv file containing IDs of all the images
        s1: float
            Folat value between 0 and 1 to split dataset into traintest 
            and holdout set. 
        s2: float
            Folat value between 0 and 1 to split traintest(as created by 
            s1 fraction split) into train and test set. 
        make_dirs: bool
            Boolean value if True then will create three directory with 
            name traindata testdata and holdoutdata data 
            in '"./Datafiles/Images/" directory
    returns: 
    --------
        Two dictionaries
            First dictionary contains two list with traintest IDs and holdout IDs
            Second dictionary contains two list with train IDs and test IDs

    """
    import pandas as pd
    import random
    import os
    allIDs = pd.read_csv(path2allLabels)

    x = allIDs['ID'].tolist()
    n=len(x)

    random.shuffle(x)
    traintest=[]
    holdout=[]
    
    j = int(n*s1)
    for i in range(j):
        holdout.append(x[i])
        x.pop(i)
    traintest = x
    
    partition1 = {'traintest':traintest,'holdout':holdout}
    print(f"Random split of Whole database IDs done -- Traintest Set have {len(traintest)} IDs")
    print(f"Holdout set have {len(holdout)} IDs")
    
    random.shuffle(traintest)
    train = []
    test= []
    
    n2 = len(traintest)
    j2 = int(n2*s2)
    for i in range(j2):
        test.append(traintest[i])
        traintest.pop(i)
    train = x
    partition2 = {'train':train,'test':test}
    print(f"Random split of Traintest IDs done -- train Set have {len(train)} IDs")
    print(f"Test set have {len(test)} IDs")
    
    if make_dirs:
        if 'Datafiles' in os.listdir('./utilities'):
            path2train = './Datafiles/Images/traindata'
            path2test = './Datafiles/Images/testdata'
            path2holdout = './Datafiles/Images/holdoutdata'
            os.makedirs(path2train,exist_ok=True)
            os.makedirs(path2test,exist_ok=True)
            os.makedirs(path2holdout,exist_ok=True)
        else:
            print('\nTHERE IS NO DIRECTORY NAMED -- datafiles-- CONTAINING ALL IMAGES AND LABELS')
    return partition1, partition2


def move_files(
                partition1,
                partition2,
                path_images = './Datafiles/Images/AllNumpyImages_211/',
                path2holdout='./Datafiles/Images/holdoutdata',
                path2train='./Datafiles/Images/traindata',
                path2test='./Datafiles/Images/testdata',
                filetype='.npy'
               ):
    '''
    This function copies all the files from source directory to holdout,
    traindata and testdata directory depending on the keys and values present 
    in given dictionary.First create dictionary using custom made function located 
    in utilities.py file. Name of the function is split_traintest()

    Arguments:
    ---------
        partition1: dict
                    Dictionary havin two keys ('traintest' and 'holdout') and 
                    respective IDs generated from split_traintest() function

        partition2: dict
                    Dictionary havin two keys ('train' and 'test') and 
                    respective IDs generated from split_traintest() function

        path_images: str
                    Path to all the CT images

        path2holdout: str
                    Path to store holdout files

        path2train: str
                    Path to store train files

        path2test: str
                    Path to store test files
    returns: 
    --------
        nothing

    '''
    import os
    import shutil
    IDs2move2holdout = partition1['holdout']
    IDs2move2test = partition2['test']
    IDs2move2train = partition2['train']
    
    for k in IDs2move2holdout:
        src_holdout = path_images+str(k)+filetype
        if str(k)+filetype not in os.listdir(path2holdout):
            shutil.copy(src_holdout,path2holdout)
            print(f'File: {src_holdout} moved to {path2holdout}')
        else:
            print(f'{k} File already exisit')
    print(f'{len(IDs2move2holdout)} files copied to {path2holdout}')

    for i in IDs2move2train:
        src_train = path_images+str(i)+filetype
        if str(i)+filetype not in os.listdir(path2train):
            shutil.copy(src_train,path2train)
            print(f'File: {src_train} moved to {path2train}')
        else:
            print(f'{i} File already exisit')
    print(f'{len(IDs2move2train)} files copied to {path2train}')
    for j in IDs2move2test:
        src_test = path_images+str(j)+filetype
        if str(j)+filetype not in os.listdir(path2test):
            shutil.copy(src_test,path2test)
            print(f'File: {src_test} copied to {path2test}')        
        else:
            print(f'{j} File already exisit')
    print(f'{len(IDs2move2test)} files moved to {path2test}') 


def getlabels(
            path_test = "./Datafiles/Images/testdata/",
            path_train= "./Datafiles/Images/traindata/",
            path_holdout='./Datafiles/Images/holdoutdata/',
            path_labels = 'Datafiles/Labels/labels_211.csv',
            target='Overall_CR_Y0_N1',
            clin_feat=['Sex_M0_F1','CC_Length_cm','Chemo_new']
            ):
    '''
    This function generates labels from training, test and holdout images 
    assuming that image names contains repective ID followed by .npy extension
    example: 111111.npy

    Arguments:
    ----------

        path_test: str
                    path to test images. Default path is 
                    "./Datafiles/Images/testdata/"

        path_train: str
                    path to test images. Default path is 
                    "./Datafiles/Images/traindata/"

        path_holdout: str
                    Path to holdout images. Default is 
                    "./Datafiles/Images/holdoutdata/"  

        path_labels: str
                    path to csv files containing outcome data 
                    with one of the header as "Overall_CR_Y0_N1" 
                    indicating overall pathological response. 

        target: str
                string representing target column in a csv file
                default is "Overall_CR_Y0_N1"

        clin_feat: list
                A list of strings representing clinical features 
                in a column of a csv file. Default is 
                ['Sex_M0_F1','CC_Length_cm','Chemo_new']

    Returns: 
    -------
        List and Dictionries 
            It returns 3 List (trainIDs,testIDs,holdoutIDs) 
            these list contains the IDs of respective set

            It also returns 6 Dictionaries 
            (train_labels,test_labels,holdout_labels,
            train_clinfeat,test_clinfeat and holdout_clinfeat)
            these dictionaries have  key as ID and value as 3 clinical features 
            for each respective ID

    Usage: 
    -----
    
        trainIDs,testIDs,holdoutIDs,
        train_labels,test_labels,holdout_labels,
        train_clinfeat,test_clinfeat,holdout_clinfeat = getlables() 
        This will run the function using default values for all the arguments

    '''
    import os
    import pandas as pd 
    testIDs = []
    for ID in os.listdir(path_test):
        testIDs.append(str(ID[:-4])+'.npy')
    trainIDs = []
    for ID in os.listdir(path_train):
        trainIDs.append(str(ID[:-4])+'.npy')
    holdoutIDs = []
    for ID in os.listdir(path_holdout):
        holdoutIDs.append(str(ID[:-4])+'.npy')

    df = pd.read_csv(path_labels)

    holdout_labels = {}
    holdout_clinfeat = {}

    for i in range(len(holdoutIDs)):
        for j in range(len(df)):
            if int(holdoutIDs[i][:-4]) == df['ID'].iloc[j]:
                holdout_labels.update({str(df['ID'].iloc[j])+'.npy':df[target].iloc[j]})
                holdout_clinfeat.update({str(df['ID'].iloc[j])+'.npy':df[clin_feat].iloc[j]})
    print(f"Holdout Labels added to respective IDs for {len(holdout_labels)} patients") 

    test_labels = {}
    test_clinfeat = {}

    for i in range(len(testIDs)):
        for j in range(len(df)):
            if int(testIDs[i][:-4]) == df['ID'].iloc[j]:
                test_labels.update({str(df['ID'].iloc[j])+'.npy':df[target].iloc[j]})
                test_clinfeat.update({str(df['ID'].iloc[j])+'.npy':df[clin_feat].iloc[j]})
    print(f"Test Labels added to respective IDs for {len(test_labels)} patients")

    train_labels = {}
    train_clinfeat = {}
    for i in range(len(trainIDs)):
        for j in range(len(df)):
            if int(trainIDs[i][:-4]) == df['ID'].iloc[j]:
                train_labels.update({str(df['ID'].iloc[j])+'.npy':df[target].iloc[j]})
                train_clinfeat.update({str(df['ID'].iloc[j])+'.npy':df[clin_feat].iloc[j]})
    print(f"Train Labels added to respective IDs for {len(train_labels)} patients")

    
    return (trainIDs,testIDs,holdoutIDs,
            train_labels,test_labels,holdout_labels,
            train_clinfeat,test_clinfeat,holdout_clinfeat)

def shuffle_traintest(
                    path2train = './Datafiles/Images/traindata',
                    path2test = './Datafiles/Images/testdata',
                    path_images = './Datafiles/Images/AllNumpyImages_211/',
                    s=0.15
                    ):
    """

    This function shuffles the data. 
    It mixes the datat in training set and test set 
    and then shuffle them folowwed by again randomly splitting the data
    into trainset and test set.
    This process does not impact the holdout set

    Argumnets:
    ----------
    path2train: str
                path to folder containing training images. 
                default './Datafiles/Images/traindata'

    path2test: str
                path to folder containing test images.
                default './Datafiles/Images/testdata'

    path_images: str
                path to folder containing all the images. 
                This is the original folder from which 
                training set and test set was created.

    s: float
        float value which defines the split ratio for training and test set

    Returns:
    -------- 
    Print the new length of training and test set. Holdout set remains the same


    """

    import random
    import os
    import shutil

    IDtrain = os.listdir(path2train)
    IDtest = os.listdir(path2test)
    IDtraintest = IDtrain + IDtest
    random.shuffle(IDtraintest)

    train = []
    test= []

    n = len(IDtraintest)
    j = int(n*s)
    for i in range(j):
        test.append(IDtraintest[i])
        IDtraintest.pop(i)
    train = IDtraintest
    partition = {'train':train,'test':test}
    IDs2move2test = partition['test']
    IDs2move2train = partition['train']
    print(f"Random re-split of Traintest IDs done -- train Set have {len(train)} IDs")
    print(f"Test set have {len(test)} IDs")

    try:
        if 'traindata' and 'testdata' in os.listdir('./Datafiles/Images/'):
            shutil.rmtree('./Datafiles/Images/traindata/', ignore_errors=True)
            shutil.rmtree('./Datafiles/Images/testdata/', ignore_errors=True)
            os.makedirs(path2train,exist_ok=True)
            os.makedirs(path2test,exist_ok=True)
        else:
            os.makedirs(path2train,exist_ok=True)
            os.makedirs(path2test,exist_ok=True)
        
    except (IOError, OSError) as e:
        ans = int(input("Please close the directory and try again. To try again enter 1:  "))
        if ans == 1:
            if 'traindata' in os.listdir('./Datafiles/Images/'):
                shutil.rmtree('./Datafiles/Images/traindata/', ignore_errors=True)
            if 'testdata' in os.listdir('./Datafiles/Images/'):
                shutil.rmtree('./Datafiles/Images/testdata/', ignore_errors=True)
            
            os.makedirs(path2train,exist_ok=True)
            os.makedirs(path2test,exist_ok=True)

    for i in IDs2move2train:
        src_train = path_images+i
        if i not in os.listdir(path2train):
            shutil.copy(src_train,path2train)
            #print(f'File: {src_train} moved to {path2train}')
        else:
            print(f'{i} File already exisit')
    print(f'{len(IDs2move2train)} files copied to {path2train}')
    for j in IDs2move2test:
        src_test = path_images+j
        if j not in os.listdir(path2test):
            shutil.copy(src_test,path2test)
            #print(f'File: {src_test} copied to {path2test}')        
        else:
            print(f'{j} File already exisit')
    print(f'{len(IDs2move2test)} files moved to {path2test}') 

def check_classes(train_labels,
                  test_labels,
                  holdout_labels):
    """
    Function to check the balance in classes

    Arguments:
    ----------
    train_labels: dictionary
                    dictionary containing labels for training set. 
                    This is the Same dictionary as returned by getlabels function
                    present in utilities.py file

    test_labels: dictionary
                    dictionary containing labels for test set. 
                    This is the Same dictionary as returned by getlabels function
                    present in utilities.py file

    holdout_set: dictionary
                    dictionary containing labels for holdout set. 
                    This is the Same dictionary as returned by getlabels function
                    present in utilities.py file

    Returns:
    --------
    Print the percentage of each class within respective set

    """
    cr_tr=0
    pr_tr=0
    cr_te=0
    pr_te=0
    cr_ho=0
    pr_ho=0

    for i in holdout_labels.values():
        if i == 0:
            cr_ho+=1
        if i == 1:
            pr_ho+=1
    print(f'Holdout set: Patient with CR: {cr_ho} and patient with PR: {pr_ho}')

    for i in train_labels.values():
        if i == 0:
            cr_tr+=1
        if i == 1:
            pr_tr+=1
    print(f'Training set: Patient with CR: {cr_tr} and patient with PR: {pr_tr}')

    for i in test_labels.values():
        if i == 0:
            cr_te+=1
        if i == 1:
            pr_te+=1
    print(f'Test set: Patient with CR: {cr_te} and patient with PR: {pr_te}')

def check_datagen_images(img,index=0):
    
    '''
    Function to display the CTscan images of particular patient as 
    generated by imagedatagenerator
    Arguments:
    -----------
    index: int 
            value to select the CTscan of one patient to display.
    img: numpy array as returned by imagedatagenerator of size(index,64,64,64,1)
            This array represent the CTscsn images of batch of patients
            as generated by imagedatagenerator. 

    Returns:
    --------
    Display the images

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    my_img = np.rot90(img[index])

    fig, axes = plt.subplots(nrows=8, ncols=8,figsize=(24,24),sharex='col', sharey='row')
    fig.suptitle(f'CT Scan Images for: {ID}',fontsize=18,fontweight=5)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.3)

    for ax, i in zip(axes.flatten(), range(len(my_img))):
        ax.imshow(my_img[:,:,i],cmap='gray')
        ax.set(title=f'Slice no:{i}')
        ax.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
       

def check_numpy_images(ID,img_path='./Datafiles/Images/AllNumpyImages_211/'):
    
    '''
    Function to display the CTscan images of particular 
    patient (for numpy images of size 64,64,64)

    Arguments:
    -----------
    ID: int 
        CRNumber of the patient whose CTscan is to be displayed.

    img_path: str
                Path to folder containing numpy images for patient 
                with input ID. numpy images should have shape of (64,64,64)                                        
    Returns:
    --------
    Display the images

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    list_files = os.listdir(img_path)
    for i in list_files:
        if i == str(ID) + '.npy':
            img = np.load(os.path.join(img_path,str(ID) + '.npy'))
            my_img = np.rot90(img)
    fig, axes = plt.subplots(nrows=8, ncols=8,figsize=(24,24),sharex='col', sharey='row')
    fig.suptitle(f'CT Scan Images for: {ID}',fontsize=18,fontweight=5)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.3)

    for ax, i in zip(axes.flatten(), range(len(my_img))):
        ax.imshow(my_img[:,:,i],cmap='gray')
        ax.set(title=f'Slice no:{i}')
        ax.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

def Qmetric(va,vl,ta,tl):
    '''
    This Function generate a scalar value to be passed on to bayesian-optimizer. 
    Bayesian-optimizer uses this value for hyperparameter optimization. Purpose is 
    to create a value which incorporates all the component of model efficacy.
    Argument:
    ---------
            va: validation accuracy
            vl: validation loss
            ta: training accuracy
            tl: training loss

    Return:
    -------
    Float value representing the overall performance of the model

    '''
    import numpy as np
    max_val = 4215 # maximum possible value with va and ta = 0.999 and vland tl = 0.01
    try:
        if vl is not np.nan:
            val = (va*2000)+(5/vl)+(ta*1500)+(2.5/tl)
            val = (val/max_val)*100 # realtive value in percentage
            if va > 0.70:
                val = val*(1/np.exp(abs(ta-va)*2)) - (np.exp(abs((vl-tl))*10) - 1)
            else:
                val = val/2
        else:
            val = (va*2000)+(ta*1500)
            val = (val/max_val)*100 # realtive value in percentage
            val = val*(1/np.exp(abs(ta-va)*4))
        
    except ZeroDivisionError as e:
        val = (va*10)+(ta*7)
        val = val*(1/np.exp(abs(ta-va)))
        print(f'QMetric was invalid because of following error: {e}')
        print('Returning modified value after omission of training and validation loss')
    if val < 0:
        return 0
    else:
        return val


def showimg(img):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(nrows=8, ncols=8,figsize=(20,20),sharex='col', sharey='row')
    fig.suptitle(f'CT Scan Images',fontsize=36,fontweight=5,y=0.92)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.3)

    for ax, i in zip(axes.flatten(), range(img.shape[2])):
        ax.imshow(np.rot90(img[:,:,i]),cmap='gray')
        ax.set(title=f'Slice no:{i+1}')


def editnumpyimg(ID,path2imgdir = '../imagedata/workingOutput',is_edit=False,edit_value=0.3,contrast='low',
                show_imgages=False):

    import os
    import numpy as np
    from utilities import utilities

    listimgs = os.listdir(path2imgdir)
    path2img = os.path.join(path2imgdir,str(ID)+'.npy')

    curr_img = np.load(path2img)
    if is_edit:
        if contrast == 'low':
            curr_img = np.where(curr_img < edit_value,curr_img,curr_img.min())
        if contrast == 'high':
            curr_img = np.where(curr_img > edit_value,curr_img,curr_img.max())
        if show_imgages:
            utilities.showimg(curr_img)
    else:
        if show_imgages:
            utilities.showimg(curr_img)
    return curr_img

def view_edit_save(IDs,i,action=1,edit_val=.12):
    ans = int(input(f'You are going to view or edit {IDs[i]}.\nAre you sure - For YES press 1 for NO press 2: \n\n'))
    if ans == 1:
        myimg_orig = editnumpyimg(IDs[i],is_edit=False,edit_value=edit_val,contrast='low',show_imgages=False)
        print(f'\nMin Value: {myimg_orig.min()}, Max Value: {myimg_orig.max()}')
        if action == 1:
            myimg_orig = editnumpyimg(IDs[i],is_edit=False,edit_value=edit_val,contrast='low',show_imgages=True)
            return myimg_orig
            
        if action == 2:
            img_low = editnumpyimg(IDs[i],is_edit=True,
                                 edit_value=myimg_orig.max() - edit_val,
                                 contrast='low',show_imgages=True)
            print(f'\nAfter Edit Low --- Min Value: {img_low.min()}, Max Value: {img_low.max()}')
            return img_low
        if action == 3:
            img_high = editnumpyimg(IDs[i],is_edit=True,
                                 edit_value=myimg_orig.min() + edit_val,
                                 contrast='high',show_imgages=True)
            print(f'\nAfter Edit High --- Min Value: {img_high.min()}, Max Value: {img_high.max()}')
            return img_high   

def save_images(IDs, img, path2save='./FinalNumpyImages32x32x64/',i=0):
    import os
    import numpy as np
    files = os.listdir(path2save)
    for k,file in enumerate(files):
        file = IDs[i] + '.npy'
        file1 = IDs[i] +'_1'+'.npy'
        file2 = IDs[i] +'_2'+'.npy'
        file3 = IDs[i] +'_3'+'.npy'
        file4 = IDs[i] +'_4'+'.npy'
    if (file in files) and (file1 in files) and (file2 in files) and (file3 in files) and (file4 in files):
        path2save = './FinalNumpyImages32x32x64/'+ IDs[i] +'_5'+'.npy'
        np.save(path2save,img)
        print('found five: '+ IDs[i]+'.npy ' + IDs[i] +'_1'+'.npy' + IDs[i] +'_2'+'.npy' + IDs[i] +'_3'+'.npy'+ IDs[i] +'_4'+'.npy')
        print(f'\nFile saved at {path2save}')
    elif (file in files) and (file1 in files) and (file2 in files)and (file3 in files):
        path2save = './FinalNumpyImages32x32x64/'+ IDs[i] +'_4'+'.npy'
        np.save(path2save,img)
        print('found Four: '+ IDs[i]+'.npy ' + IDs[i] +'_1'+'.npy' + IDs[i] +'_2'+'.npy'+ IDs[i] +'_3'+'.npy')
        print(f'\nFile saved at {path2save}')
    elif (file in files) and (file1 in files) and (file2 in files):
        path2save = './FinalNumpyImages32x32x64/'+ IDs[i] +'_3'+'.npy'
        np.save(path2save,img)
        print('found three: '+ IDs[i]+'.npy ' + IDs[i] +'_1'+'.npy' + IDs[i] +'_2'+'.npy')
        print(f'\nFile saved at {path2save}')
    elif (file in files) and (file1 in files):
        path2save = './FinalNumpyImages32x32x64/'+ IDs[i] +'_2'+'.npy'
        np.save(path2save,img)
        print('found Two: '+ IDs[i]+'.npy ' + IDs[i] +'_1'+'.npy')
        print(f'\nFile saved at {path2save}')
    elif file in files:
        path2save = './FinalNumpyImages32x32x64/'+IDs[i] +'_1'+'.npy'
        np.save(path2save,img)
        print('found one: '+ IDs[i]+'.npy')
        print(f'\nFile saved at {path2save}')
    else:
        path2save = './FinalNumpyImages32x32x64/'+IDs[i]+'.npy'
        np.save(path2save,img)
        print('found none')
        print(f'\nFile saved at {path2save}')


def genlabels(dict1,dict2,
            path_labels = 'Datafiles/Labels/labels_211.csv',
            target='Overall_CR_Y0_N1',
            clin_feat=['Sex_M0_F1','CC_Length_cm','Chemo_new']
            ):
    '''
    This function generates labels from two dictionaries containing 
    traintest, holdout, train and test IDs generated by split_traintest function.

    Arguments:
    ----------

        dict1: Dictionary containing IDs for Traintest set and holdout set
        dict2: Dictionary containing IDs for tain and test set

        path_labels: str
                    path to csv files containing outcome data 
                    with one of the header as "Overall_CR_Y0_N1" 
                    indicating overall pathological response. 

        target: str
                string representing target column in a csv file
                default is "Overall_CR_Y0_N1"

        clin_feat: list
                A list of strings representing clinical features 
                in a column of a csv file. Default is 
                ['Sex_M0_F1','CC_Length_cm','Chemo_new']

    Returns: 
    -------
        List and Dictionries 
            It returns 3 List (trainIDs,testIDs,holdoutIDs) 
            these list contains the IDs of respective set

            It also returns 6 Dictionaries 
            (train_labels,test_labels,holdout_labels,
            train_clinfeat,test_clinfeat and holdout_clinfeat)
            these dictionaries have  key as ID and value as 3 clinical features 
            for each respective ID

    Usage: 
    -----
    
        trainIDs,testIDs,holdoutIDs,
        train_labels,test_labels,holdout_labels,
        train_clinfeat,test_clinfeat,holdout_clinfeat = getlables() 
        This will run the function using default values for all the arguments

    '''
    import os
    import pandas as pd 
    testIDs = []
    for i in dict2['test']:
        testIDs.append(str(i)+'.npy')
    trainIDs = []
    for i in dict2['train']:
        trainIDs.append(str(i)+'.npy')
    holdoutIDs = []
    for i in dict1['holdout']:
        holdoutIDs.append(str(i)+'.npy')

    df = pd.read_csv(path_labels)

    holdout_labels = {}
    holdout_clinfeat = {}

    for i in range(len(holdoutIDs)):
        for j in range(len(df)):
            if int(holdoutIDs[i][:-4]) == df['ID'].iloc[j]:
                holdout_labels.update({str(df['ID'].iloc[j])+'.npy':df[target].iloc[j]})
                holdout_clinfeat.update({str(df['ID'].iloc[j])+'.npy':df[clin_feat].iloc[j]})
    print(f"Holdout Labels added to respective IDs for {len(holdout_labels)} patients") 

    test_labels = {}
    test_clinfeat = {}

    for i in range(len(testIDs)):
        for j in range(len(df)):
            if int(testIDs[i][:-4]) == df['ID'].iloc[j]:
                test_labels.update({str(df['ID'].iloc[j])+'.npy':df[target].iloc[j]})
                test_clinfeat.update({str(df['ID'].iloc[j])+'.npy':df[clin_feat].iloc[j]})
    print(f"Test Labels added to respective IDs for {len(test_labels)} patients")

    train_labels = {}
    train_clinfeat = {}
    for i in range(len(trainIDs)):
        for j in range(len(df)):
            if int(trainIDs[i][:-4]) == df['ID'].iloc[j]:
                train_labels.update({str(df['ID'].iloc[j])+'.npy':df[target].iloc[j]})
                train_clinfeat.update({str(df['ID'].iloc[j])+'.npy':df[clin_feat].iloc[j]})
    print(f"Train Labels added to respective IDs for {len(train_labels)} patients")

    
    return (trainIDs,testIDs,holdoutIDs,
            train_labels,test_labels,holdout_labels,
            train_clinfeat,test_clinfeat,holdout_clinfeat)