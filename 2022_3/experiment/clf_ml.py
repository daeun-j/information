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

files = glob.glob('./sort/network_v2/result_0506/Network_10000_0.9_epo0_hz.csv')
# files = files[:3]

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import pandas as pd
from sklearn.metrics import recall_score, precision_score, accuracy_score

def evaluate_class(y, y_hat):
    # confusion_matrix(y, y_hat, labels=[0, 1])
    # confusion_matrix(y, y_hat, labels=[0, 1, 2])

    #print(classification_report(y, y_hat))
    print("y, y_hat", y, y_hat)
    #print("Accracy {}  | macro precision {}| macro recall {}".format(accuracy_score(y, y_hat),precision_score(y, y_hat, average='macro'), recall_score(y, y_hat, average='macro'))
    return accuracy_score(y, y_hat, normalize=False), precision_score(y, y_hat, average='macro'), recall_score(y, y_hat, average='macro')



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
        #  45874
        tree = XGBClassifier(seed=200, use_label_encoder=False, eval_metric='logloss')
        x_train, x_test, y_train, y_test = train_test_split(raw,  labels, stratify= labels, shuffle=True, random_state=34, test_size=0.1)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        result = evaluate_class(y_test, y_pred)
        print('{} | raw | h | {} '.format(file, result))
        print('{} | raw | h | {} '.format(file, result), file=f)
        # f.write('{} | raw | h | {} '.format(file, result))

        del tree, x_train, x_test, y_train, y_test

        clf = XGBClassifier(seed=202, use_label_encoder=False, eval_metric='logloss')
        x_train, x_test, y_train, y_test = train_test_split(raw,  labels_u, stratify= labels_u, shuffle=True, random_state=34, test_size=0.1)
        print(len(x_train), len(x_test), len(y_train), len(y_test))

        clf.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = clf.predict(x_test)
        result = evaluate_class(y_test, y_pred)
        print('{} | raw | z | {}'.format(file, result))
        print('{} | raw | z | {} '.format(file, result), file=f)
        # f.write('{} | raw | z | {} '.format(file, result))

        del clf, x_train, x_test, y_train, y_test


        tree = XGBClassifier(seed=2002, use_label_encoder=False,eval_metric='logloss')
        x_train, x_test, y_train, y_test = train_test_split(raw,  labels_u, stratify= labels_u, shuffle=True, random_state=34, test_size=0.1)

        print(len(x_train), len(x_test), len(y_train), len(y_test) )

        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        result = evaluate_class(y_test, y_pred)
        print('{} | meta | h | {} '.format(file, result))
        print('{} | meta | h | {} '.format(file, result), file=f)
        # f.write('{} | meta | h | {} '.format(file, result))
        del tree, x_train, x_test, y_train, y_test

        tree = XGBClassifier(seed=22, use_label_encoder=False,eval_metric='logloss')

        x_train, x_test, y_train, y_test = train_test_split(raw,  labels_u, stratify= labels_u, shuffle=True, random_state=34, test_size=0.1)

        print(len(x_train), len(x_test), len(y_train), len(y_test))


        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        result = evaluate_class(y_test, y_pred)
        print('{} | meta | z | {} '.format(file, result))
        print('{} | meta | z | {} '.format(file, result), file=f)

        # f.write('{} | meta | z | {} '.format(file, result))

f.close()


