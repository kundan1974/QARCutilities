import numpy as np
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(featurewise_center=False,
                               samplewise_center=False,
                               featurewise_std_normalization=False,
                               samplewise_std_normalization=False,
                               zca_whitening=False,
                               zca_epsilon=1e-06,
                               rotation_range=10,
                               width_shift_range=[-0.05, 0.05],
                               height_shift_range=[-0.05, 0.05],
                               brightness_range=None,
                               shear_range=None,
                               zoom_range=[0.9, 1.1],
                               channel_shift_range=0.0,
                               fill_mode='constant',
                               cval=0.61,
                               horizontal_flip=True,
                               vertical_flip=True,
                               rescale=0,
                               preprocessing_function=None,
                               data_format=None,
                               validation_split=0.0,
                               dtype=None
                               )

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, clinfeat,batch_size=32, dim=(64, 64, 64), n_channels=1,
                 n_classes=1, shuffle=True, isTestData=False,  factor4CR=1, factor4PR=1, isClinicalData=False,
                 all_image_path ='./utilities/Datafiles/Images/AllNumpyImages_211/'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.clinfeat = clinfeat
        self.list_repeated_ids = self.__get_repeated_list_ids_balanced(list_IDs,isTestData,labels,factor4CR,factor4PR)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.isTestData = isTestData
        self.isClinicalData = isClinicalData
        self.on_epoch_end()
        self.all_image_path  = all_image_path 

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_repeated_ids) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_one_batch = [self.list_repeated_ids[k] for k in indexes]

        # Generate data
        if self.isClinicalData:
            X, y, clinical = self.__data_generation(list_ids_one_batch)
            return (X,clinical), y
        else:
            X, y = self.__data_generation(list_ids_one_batch)
            return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_repeated_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_one_batch):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)
        clinical = np.empty(shape=(self.batch_size,3), dtype=float)

        # Generate data
        for i, ID in enumerate(list_ids_one_batch):
            # Store sample
            if self.isTestData:
                X[i,] = np.load(os.path.join(self.all_image_path , ID)).reshape(64, 64, 64, 1)
            else:
                # generates random augmented image for each
                tmp_img = np.load(os.path.join(self.all_image_path, ID))
                aug_img = image_gen.random_transform(tmp_img) # image augmentation using random transform
                #aug_img = tmp_img # bypassing random transform
                X[i,] = aug_img.reshape(64, 64, 64, 1)
                print(ID,self.labels[ID])
            # Store class
            y[i] = self.labels[ID]
            if self.isClinicalData:
                clinical[i] = self.clinfeat[ID].values  
        if self.isClinicalData:
            return X, y, clinical
        else:
            return X, y

    def __get_repeated_list_ids(self, list_ids, images_per_id):
        'Returns a new list of IDs where each ID is repeated @images_per_id times'
        list_repeated_images_ids = []
        for id in list_ids:
            list_repeated_images_ids.extend([id] * images_per_id)
        return  list_repeated_images_ids

    def __get_repeated_list_ids_balanced(self, list_ids, isTestData,labels,factor4CR,factor4PR):
        'Returns a new list of IDs where each ID is repeated @images_per_id times'
        list_repeated_images_ids1 = []
        list_repeated_images_ids2 = []
        list_repeated_images_ids = []
        if isTestData:
            for ids in list_ids:
                list_repeated_images_ids.append(ids)
            return  list_repeated_images_ids
        else:    
            for ids in list_ids:
                if labels[ids] == 0:
                    list_repeated_images_ids1.extend([ids] * factor4CR)
                else:
                    list_repeated_images_ids2.extend([ids] * factor4PR)
            list_repeated_images_ids = list_repeated_images_ids1 + list_repeated_images_ids2    
            return  list_repeated_images_ids
    
