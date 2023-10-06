import zipfile
import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class dataloader:
    """
        A class for loading and preprocessing image data from a zip file.

        Args:
            filename (str): The path to the zip file containing the dataset.
            extract_to (str): The directory where the dataset will be extracted.

        Attributes:
            filename (str): The path to the zip file.
            extract_to (str): The directory where the dataset is extracted.
            STORE_DATA (list): A list containing preprocessed image data.

        Methods:
            _unzip_file(self, filename=None, extract_to=None):
                Extracts the zip file to the specified directory.

            _extract_features(self, DIRECTORY=None):
                Reads and preprocesses images from the dataset.

            _split_dataset(self, STORE_DATA=None):
                Splits the dataset into training and testing sets.

            _shuffle_dataset(self, STORE_DATA=None):
                Shuffles the dataset randomly.

            _split_independent_dependent(self, STORE_DATA=None):
                Splits the dataset into independent features and labels.

            _reshaping(self, STORE_DATA=None):
                Reshapes the data for model compatibility.

            _create_data_loader(self, X_train=None, X_test=None, y_train=None, y_test=None):
                Creates data loaders for training and testing data.

        Usage:
            load = DataLoader(filename='c:/Users/atiku/Downloads/dataset_mnist.zip', extract_to='C:/Users/atiku/Downloads/')
    """
    def __init__(self, filename=None, extract_to=None):
        self.filename = filename
        self.extract_to = extract_to
        self._unzip_file(filename=self.filename, extract_to=self.extract_to)
        self.STORE_DATA = self._extract_features(DIRECTORY=os.path.join(self.extract_to, 'dataset'))
        X, y, TRAIN_LOADER, TEST_LOADER = self.dataset(STORE_DATA = self.STORE_DATA)
    
    def dataset(self, STORE_DATA = None):
        X, y, TRAIN_LOADER, TEST_LOADER = self._split_dataset(STORE_DATA = STORE_DATA)
        
        return X, y, TRAIN_LOADER, TEST_LOADER
    
    def _unzip_file(self, filename=None, extract_to=None):
        """
            Extracts the zip file to the specified directory.

            Args:
                filename (str): The path to the zip file.
                extract_to (str): The directory where the zip file will be extracted.
        """
        link_folder = filename
        if link_folder.split(".")[-1] == 'zip':
            print("Unzipping is in progress. It will take some time, so please be patient.\n")
            with zipfile.ZipFile(link_folder, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            raise Exception("File should be in zip format")

    def _extract_features(self, DIRECTORY=None):
        """
            Reads and preprocesses images from the dataset.

            Args:
                DIRECTORY (str): The directory containing the dataset.

            Returns:
                STORE_DATA (list): A list containing preprocessed image data.
        """
        CATEGORIES = [str(i) for i in range(0, 10)]
        STORE_DATA = []

        for category in CATEGORIES:
            FOLDER_PATH = os.path.join(DIRECTORY, category, category)

            for IMAGE in os.listdir(FOLDER_PATH):
                IMAGE_PATH  = os.path.join(FOLDER_PATH, IMAGE)
                
                # Read the image as grayscale (1 channel)
                grayscale_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
                # Resize the grayscale image to (28, 28)
                grayscale_image = cv2.resize(grayscale_image, (28, 28))
                # Expand dimensions to (28, 28, 1)
                grayscale_image = np.expand_dims(grayscale_image, axis=-1)
                
                IMAGE_LABEL = CATEGORIES.index(category)
                STORE_DATA.append([grayscale_image, IMAGE_LABEL])

            print("Folder {} is completed.".format(category))
            print("_" * 30)

        return STORE_DATA
    
    def _split_dataset(self, STORE_DATA = None):
        """
            Splits the dataset into training and testing sets.

            Args:
                STORE_DATA (list): A list containing preprocessed image data.

            Returns:
                X (numpy.ndarray): The independent features.
                y (numpy.ndarray): The corresponding labels.
                TRAIN_LOADER (torch.utils.data.DataLoader): DataLoader for training data.
                TEST_LOADER (torch.utils.data.DataLoader): DataLoader for testing data.
        """
        self._shuffle_dataset(STORE_DATA = STORE_DATA)
        
        print("\nLength of Train data # {} ".format(len(STORE_DATA)),'\n')
        
        X, y = self._split_indpendent_dependent(STORE_DATA = STORE_DATA)
        
        X = X/255
        
        X = self._reshaping(X = X)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X,\
                                                            y,\
                                                            test_size = 0.25,\
                                                            random_state = 42)

        # Convert the train and test into Float with respect to torch
        X_train = torch.tensor(data  = X_train,\
                               dtype = torch.float32)

        X_test  = torch.tensor(data  = X_test,\
                               dtype = torch.float32)

        print("X_train shape # {} ".format(X_train.shape), '\n')
        print("y_train shape # {} ".format(y_train.shape), '\n')
        print("X_test shape  # {} ".format(X_test.shape), '\n')
        print("y_test shape  # {} ".format(y_test.shape), '\n')
        
        print("_"*60, '\n')
        
        train_loader, test_loader = self._create_data_loader(X_train = X_train,
                                                             y_train = y_train,
                                                             X_test  = X_test,
                                                             y_test  = y_test)
        
        return X, y, train_loader, test_loader
    
    def _shuffle_dataset(self, STORE_DATA = None):
        """
            Shuffles the dataset randomly.

            Args:
                STORE_DATA (list): A list containing preprocessed image data.

            Returns:
                None
        """
        random.shuffle(STORE_DATA)
    
    def _split_indpendent_dependent(self, STORE_DATA = None):
        """
            Splits the dataset into independent features and labels.

            Args:
                STORE_DATA (list): A list containing preprocessed image data.

            Returns:
                X (numpy.ndarray): The independent features.
                y (numpy.ndarray): The corresponding labels.
        """
        
        X = []
        y = []

        for (independent, dependent) in STORE_DATA:
            X.append(independent)
            y.append(dependent)
            
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def _reshaping(self, X = None):
        """
            Reshapes the data for model compatibility.

            Args:
                STORE_DATA (list): A list containing preprocessed image data.

            Returns:
                X (numpy.ndarray): The reshaped data.
        """
        CHANNEL = 1
        HEIGHT  = 28
        WIDTH   = 28

        X = X.reshape(X.shape[0], CHANNEL, HEIGHT, WIDTH)
        X = X.reshape(X.shape[0], -1)
        
        return X
    
    def _create_data_loader(self, X_train = None, X_test = None, y_train = None, y_test = None):
        """
            Creates data loaders for training and testing data.

            Args:
                X_train (torch.Tensor): Training data.
                X_test (torch.Tensor): Testing data.
                y_train (numpy.ndarray): Training labels.
                y_test (numpy.ndarray): Testing labels.

            Returns:
                TRAIN_LOADER (torch.utils.data.DataLoader): DataLoader for training data.
                TEST_LOADER (torch.utils.data.DataLoader): DataLoader for testing data.
        """
        BATCH_SIZE = 64

        TRAIN_LOADER = DataLoader(dataset = list(zip(X_train, y_train)),\
                                batch_size = BATCH_SIZE,\
                                shuffle = True)

        TEST_LOADER  = DataLoader(dataset = list(zip(X_test, y_test)),\
                                batch_size = BATCH_SIZE,\
                                shuffle = True)

        print("Batch size of Train # {} ".format(TRAIN_LOADER.batch_size), '\n')
        print("Batch size of Test  # {} ".format(TEST_LOADER.batch_size), '\n')

        print("_"*60, '\n')

        # Extract the data and label
        train_data, train_label = next(iter(TRAIN_LOADER))
        test_data, test_label   = next(iter(TEST_LOADER))

        print("Train data with single batch_size  # {} ".format(train_data.shape), '\n')
        print("Train label with single batch_size # {} ".format(train_label.shape), '\n')
        print("Test data with single batch_size   # {} ".format(test_data.shape), '\n')
        print("Test label with single batch_size  # {} ".format(test_label.shape))
        
        return TRAIN_LOADER, TEST_LOADER
        
        

if __name__ == "__main__":
    load = dataloader(filename='c:/Users/atiku/Downloads/dataset_mnist.zip', extract_to='C:/Users/atiku/Downloads/')
