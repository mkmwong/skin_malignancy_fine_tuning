import os
import cv2
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

random.seed(50)
RESIZED = 100
BATCH_SIZE = 100
COMPONENTS = 50

#### Function to import all data and return 4 numpy arrays,
#### X_train, X_test, y_train, y_test
def import_ISIC():
    #### Get all name of input images
    print("Importing input files...")
    ipath = "ISIC_2019_Training_Input"
    all_f = os.listdir(ipath) 
    files = [ fname for fname in all_f if fname.endswith('.jpg')]

    #### These two lines are only used for random sampling for testing ####
    pos = random.sample(range(0,25331),500)
    files = [files[i] for i in pos]
    ##################################################

    #### Importing image and resizing to desired size, scaling to {0,1}
    f_name = [fname.split(".")[0] for fname in files]
    imgs=None
    count = 0 
    for image in files:
        im = cv2.imread(ipath+"/"+image)
        if count % 100 == 0:
            print(str(count) + " image files are imported...")
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
    y[(y == 'NV') | (y == 'AK') | (y == 'BKL') | (y == 'DF') | (y == 'VASC')] = 0
    y[(y == 'MEL') | (y == 'SCC') | (y == 'BCC')] = 1
    y=y.astype('int')

    #### Splitting Training/test set
    print("Train/test set splitting...")
    X_train, X_test, y_train, y_test = train_test_split( imgs, y, test_size=0.25, random_state=42)
    return (X_train, X_test, y_train, y_test)

#### Making a plot of number of PCs against Cumulative variance explained
#### To help determine how many PCs are needed to build the model
def dimension_reduction_evaluation(X_train, n_comp):
    pca = IncrementalPCA(batch_size=BATCH_SIZE, n_components = n_comp)
    pca.fit(X_train)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number Of Components')
    plt.ylabel('Cumulative Variance Explained');
    plt.savefig('PCAfig.png')

#### Perform actual dimension reduction
def dimension_reduction(X_train, X_test, n_comp):
    pca = IncrementalPCA(batch_size=BATCH_SIZE, n_components = n_comp)
    x_train_out = pca.fit_transform(X_train)
    x_test_out = pca.transform(X_test)
    return(x_train_out, x_test_out)

#### class to output all metrics of a model
class Classifier_Performance:

    def __init__(self, clf, name):
        self.clf = clf
        self.name = name

    #### Training the model
    def train_model(self, X_train, y_train):
        print("Training the model...")
        self.clf.fit(X_train, y_train)
        return

    #### Ploting AUC to show model performance and save image as PNG
    def save_auc(self, X_test, y_test):
        print("Exporting AUC curve...")
        train_predictions_proba = self.clf.best_estimator_.predict_proba(X_test)[:,1]
        n_classes=2
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = roc_curve(y_test, train_predictions_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Random Forest AUC')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(self.name + 'AUC.png')
        return

    #### Saving classification report and confusion matrix as csv 
    def save_performance(self, X_test, y_test):
        print("Exporting classification report and confusion matrix...")
        train_predictions = self.clf.best_estimator_.predict(X_test)
        class_rep = classification_report(y_test, train_predictions, zero_division=0,output_dict=True)
        conf_mtx = confusion_matrix(y_test, train_predictions)
        class_rep_df = pd.DataFrame(class_rep).transpose()
        class_rep_df.to_csv(self.name + "_classification_report.csv",index=False)
        np.savetxt(self.name + "confusion_matrix.csv", conf_mtx, delimiter=",")
        return
    #### Running all the steps above and export the final model
    def run(self, X_train, y_train, X_test, y_test):
        self.train_model(X_train, y_train)
        self.save_performance(X_test, y_test)
        self.save_auc(X_test, y_test)
        print("Exporting the model...")
        dump(self.clf.best_estimator_, self.name + '.joblib')
        print("Done!")
        return

### Importing and processing data
X_train, X_test, y_train, y_test = import_ISIC() 
dimension_reduction(X_train, X_test, 50)

#### Testing random forest
print("Testing random forest...")
N_ESTIMATORS_OPTIONS = [50,100,150,200]
param_grid = [
    {
        'n_estimators' : N_ESTIMATORS_OPTIONS,
        'criterion' : ['gini', 'entropy']
    }
]
mod = Classifier_Performance(GridSearchCV(RandomForestClassifier(class_weight={0:1,1:10})
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'), "Random_forest")
mod.run(X_train, y_train, X_test, y_test)

#### Testing logistic regression
print("Testing logistic regression...")
C_OPTIONS = [0.1,1,10]
param_grid = [
    {
        'C': C_OPTIONS
    }
]
mod = Classifier_Performance(GridSearchCV(LogisticRegression(max_iter=10000,class_weight={0:1,1:10})
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'), "Logistic_regression")
mod.run(X_train, y_train, X_test, y_test)

#### Testing naive bayes
print("Testing naive bayes...")
VAR_SM = [1e-9, 1e-10, 1e-11]
param_grid = [
    {
        'var_smoothing': VAR_SM
    }
]
mod = Classifier_Performance(GridSearchCV(GaussianNB()
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'), "Naive_bayes")
mod.run(X_train, y_train, X_test, y_test)

#### Testing knn
print("Testing k-nearest neighbor...")
N_NEIGHBORS_OPTIONS = [20,50,100]

param_grid = [
    {
        'n_neighbors' : N_NEIGHBORS_OPTIONS
    }
]
mod = Classifier_Performance(GridSearchCV(KNeighborsClassifier()
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'), "KNN")
mod.run(X_train, y_train, X_test, y_test)

#### Testing decision tree
print("Testing decision tree...")
param_grid = [
    {
        'criterion' : ['gini', 'entropy']
    }
]
mod = Classifier_Performance(GridSearchCV(DecisionTreeClassifier(class_weight={0:1,1:10})
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'), "Decision_tree")
mod.run(X_train, y_train, X_test, y_test)

#### Testing linear SVM
print("Testing linear SVM...")
C_OPTIONS = [0.1,1]
g_OPTIONS = [0.1,1]
param_grid = [
    {
        'C': C_OPTIONS,
        'gamma': g_OPTIONS,
    }
]
mod = Classifier_Performance(GridSearchCV(SVC(kernel='linear',class_weight={0:1,1:10}, probability=True)
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'), "Linear_SVM")
mod.run(X_train, y_train, X_test, y_test)

#### Testing rbf SVM
print("Testing rbf SVM...")
C_OPTIONS = [0.1,1]
g_OPTIONS = [0.1,1]
param_grid = [
    {
        'C': C_OPTIONS,
        'gamma': g_OPTIONS,
    }
]
mod = Classifier_Performance(GridSearchCV(SVC(kernel='rbf',class_weight={0:1,1:10}, probability=True)
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'), "Rbf_SVM")
mod.run(X_train, y_train, X_test, y_test)

#### Testing Adaboost
print("Testiing Adaboost...")
lr_OPTIONS = [0.01,0.1,1]
n_OPTIONS = [20,50,100]

param_grid = [
    {
        'n_estimators': n_OPTIONS,
        'learning_rate': lr_OPTIONS,
    }
]
mod = Classifier_Performance(GridSearchCV(AdaBoostClassifier()
    ,n_jobs=-1, param_grid=param_grid, scoring='f1_macro'), "Adaboost")
mod.run(X_train, y_train, X_test, y_test)

