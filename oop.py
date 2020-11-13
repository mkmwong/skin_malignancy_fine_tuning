import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

#### Function to import all data and return 4 numpy arrays,
#### X_train, X_test, y_train, y_test
def import_ISMC():
    random.seed(50)
    RESIZED = 100
    BATCH_SIZE = 100
    COMPONENTS = 50

    #### Get all name of input images
    print("Importing input files...")
    ipath = "ISIC_2019_Training_Input"
    all_f = os.listdir(ipath) 
    files = [ fname for fname in all_f if fname.endswith('.jpg')]

    #### These two lines are only used for random sampling for testing ####
    pos = random.sample(range(0,25331),25331)
    files = [files[i] for i in pos]
    ##################################################

    #### Importing image and resizing to desired size, scaling to {0,1}
    f_name = [fname.split(".")[0] for fname in files]
    imgs=None
    count = 0 
    for image in files:
        im = cv2.imread(ipath+"/"+image)
        if count % 100 == 0:
            print(int(count) + " image files are imported...")
        im = cv2.resize(im, (RESIZED, RESIZED))
        im = im/255
        if imgs is None:
            imgs = im[np.newaxis,...]
        else:
            imgs = np.concatenate((imgs,im[np.newaxis,...]), axis=0)
        count = count + 1
    nrow, nx, ny, nz = imgs.shape
    imgs = imgs.reshape((nrow,nx*ny*nz))

    #### Importing metadata 
    metadata = pd.read_csv("ISIC_2019_Training_Metadata.csv")
    metadata = metadata.drop('lesion_id', axis = 1)
    #### This is only used for random sampling for testing
    metadata = metadata[metadata['image'].isin(f_name)]
    #################################################
    metadata = metadata.drop(['image'], axis = 1)

    #### Filling in missing value with mean or mode
    metadata['age_approx'] = metadata['age_approx'].fillna(metadata['age_approx'].mean())
    metadata['sex'] = metadata['sex'].fillna(metadata['sex'].mode()[0])
    metadata['anatom_site_general'] = metadata['anatom_site_general'].fillna(metadata['anatom_site_general'].mode()[0])
    metadata['anatom_site_general'] = pd.Categorical(metadata['anatom_site_general'])
    metadata['anatom_site_general'] = metadata['anatom_site_general'].cat.codes
    metadata['sex'] = pd.Categorical(metadata['sex'])
    metadata['sex'] = metadata['sex'].cat.codes
    metadata = metadata.to_numpy()

    #### Merging metadata and images
    imgs = np.concatenate((imgs, metadata), axis  =1)

    #### Importing labels as binary; 0 for benign lesion and 1 for malignant
    print("Importing labels...")
    labs = pd.read_csv("ISIC_2019_Training_GroundTruth.csv")
    labs = labs[labs['image'].isin(f_name)]
    y = labs.drop(['image'], axis = 1).idxmax(axis=1).ravel()
    y[(y == 'NV') | (y == 'AK') | (y == 'BKL') | (y == 'DF') | (y == 'VASC') |] = 0
    y[(y == 'MEL') | (y == 'SCC') | (y == 'BCC')] = 1
    y=y.astype('int')

    #### Splitting Training/test set
    print("Train/test set splitting...")
    X_train, X_test, y_train, y_test = train_test_split( imgs, y, test_size=0.25, random_state=42)
    return (X_train, X_test, y_train, y_test)


class Classifier_Performance:

    def __init__(self, clf, X_train, X_test, y_train, y_test):
        self.clf = clf
        
        def train_model(self):
            self.clf.fit()
            return

        def save_auc(self):
            return

        def save_performance(self):
            return


### testing random forest
N_ESTIMATORS_OPTIONS = [50,100,150,200]
param_grid = [
    {
        'classify__n_estimators' : N_ESTIMATORS_OPTIONS,
        'classify__criterion' : ['gini', 'entropy']
    }
]
a = Classifier_Performance(GridSearchCV(RandomForestClassifier(class_weight={0:1,1:10})
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'))

print(a)

