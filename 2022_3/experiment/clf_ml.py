from args import *
from utils import *
import glob, os
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numba

encoder = LabelEncoder()
args = parser.parse_args()
path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)
# PATH = "2022_3/experiment/sort/network/" # for python console
PATH = "./sort/network_v2/result_0506/"
output_filename = "output.csv"
datatype = 'Network_{}'.format(args.epochs)

files = glob.glob('./sort/network_v2/result_0506/Network_10000_*.csv')
# files = files[:3]

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import torch.utils.data
from torch.utils.data import DataLoader
import csv
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from utils import evaluate_class
import numpy as np
import time

# https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/
# correlation between test harness and ideal test condition

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#
# def get_models():
#     models = list()
#     # models.append(LogisticRegression())
#     # models.append(RidgeClassifier())
#     # models.append(SGDClassifier())
#     # models.append(PassiveAggressiveClassifier())
#     # models.append(KNeighborsClassifier())
#     # models.append(DecisionTreeClassifier())
#     # models.append(ExtraTreeClassifier())
#     # models.append(LinearSVC())
#     # models.append(SVC())
#     models.append(GaussianNB())
#     # models.append(AdaBoostClassifier())
#     # models.append(BaggingClassifier())
#     # models.append(RandomForestClassifier())
#     # models.append(ExtraTreesClassifier())
#     # models.append(GaussianProcessClassifier())
#     # models.append(GradientBoostingClassifier())
#     # models.append(LinearDiscriminantAnalysis())
#     # models.append(QuadraticDiscriminantAnalysis())
#     return models
#
#
# def predict_ML(model, X_train, X_test, y_train):
#     model.fit(X_train, y_train.astype(int))
#     y_pred = model.predict(X_test)
#     return y_pred.astype(int)
#
# # define test conditions
# # get the list of models to consider
# models = get_models()
# # evaluate each model
# for model in models:
#     start = time.time()
#     # evaluate model using each test condition
#     y_pred = predict_ML(model)
#
#     test_time = "{}".format(time.time()-start)
#     print(type(model).__name__ + "time :", test_time, evaluate_class(y_test, y_pred))


with open('{}{}'.format(PATH, output_filename),'w') as f:
    for i, file in enumerate (files):

        df = pd.read_csv(file)
        labels = df.iloc[:, -2].to_numpy()
        labels_u = df.iloc[:, -1].to_numpy() + labels*10

        encoder.fit(labels_u)
        labels_u = encoder.transform(labels_u)
        raw = df.iloc[:, :8].to_numpy()
        h = df.iloc[:, 8:12].to_numpy()
        z = df.iloc[:, 12:16].to_numpy()

        tree = XGBClassifier(seed=2002, use_label_encoder=False)
        x_train, x_test, y_train, y_test = train_test_split(raw,  labels, test_size=0.3)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        result = evaluate_class(y_test, y_pred)
        print('{} | raw | h | {} '.format(file, result), file=f)

        tree = XGBClassifier(seed=2002, use_label_encoder=False)
        x_train, x_test, y_train, y_test = train_test_split(raw, labels_u, test_size=0.3)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        evaluate_class(y_test, y_pred)
        print('{} | raw | z | {} '.format(file, result), file=f)


        tree = XGBClassifier(seed=2002, use_label_encoder=False)
        x_train, x_test, y_train, y_test = train_test_split(h,  labels, test_size=0.3)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        evaluate_class(y_test, y_pred)
        print('{} | meta | h | {} '.format(file, result), file=f)


        tree = XGBClassifier(seed=2002, use_label_encoder=False)
        x_train, x_test, y_train, y_test = train_test_split(z,  labels_u, test_size=0.3)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        evaluate_class(y_test, y_pred)
        print('{} | meta | z | {} '.format(file, result), file=f)


f.close()


