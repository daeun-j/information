from args import *
from utils import *
import glob, os
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numba
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


encoder = LabelEncoder()
args = parser.parse_args()
path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)
# PATH = "2022_3/experiment/sort/network/" # for python console
# PATH = "./sort/network_v2/result_0506/"
# output_filename = "output.csv"
# files = glob.glob('./sort/network_v2/result_0506/Network_10000_0.9_epo0_hz.csv')
# datatype = 'Network_{}'.format(args.epochs)
# files = glob.glob('./sort/network_v2/result_0506/simulation_21_epo20_hz.csv')


PATH = "sort/simulation_v3/result_0607/"
output_filename = "submission_{}.csv".format(args.clf)
files = glob.glob('./{}simulation_*_hz.csv'.format(PATH))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# https://hwi-doc.tistory.com/entry/%EC%9D%B4%ED%95%B4%ED%95%98%EA%B3%A0-%EC%82%AC%EC%9A%A9%ED%95%98%EC%9E%90-XGBoost
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

models = { "rf": RandomForestClassifier(),
           "knn": KNeighborsClassifier(),
           "ada": AdaBoostClassifier(),
           "qda": QuadraticDiscriminantAnalysis(),
           "xgb" : XGBClassifier(seed=200, use_label_encoder=False, eval_metric='logloss')}

print(models[args.clf])

with open('{}{}'.format(PATH, output_filename),'w') as f:
    for i, file in enumerate (files):
        totepo = file.split('_')[3]
        curepo = file.split('_')[4].split('epo')[1]

        df = pd.read_csv(file).iloc[1:, :]

        # labels = df.iloc[:, -2].to_numpy()
        # labels_u = df.iloc[:, -1].to_numpy() + labels*10

        # encoder.fit(labels_u)
        # labels_u = encoder.transform(labels_u)
        # raw = df.iloc[:, :8].to_numpy()
        # h = df.iloc[:, 8:12].to_numpy()
        # z = df.iloc[:, 12:16].to_numpy()

        labels = df.iloc[:, -2].to_numpy()
        labels_u = df.iloc[:, -1].to_numpy() + labels*10

        encoder.fit(labels_u)
        labels_u = encoder.transform(labels_u)

        encoder.fit(labels)
        labels = encoder.transform(labels)


        raw = df.iloc[:, :8].to_numpy()
        h = df.iloc[:, 8:16].to_numpy()
        z = df.iloc[:, 16:32].to_numpy()
        test_size = 0.1
        #  45874

        tree = models[args.clf]
        x_train, x_test, y_train, y_test = train_test_split(raw,  labels, stratify= labels, shuffle=True, random_state=3, test_size=test_size)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        acc, prec, recal = evaluate_class(y_test, y_pred)
        print(' raw | h | {} | {} | {} | {} | {}'.format(totepo, curepo, acc, prec, recal))
        print(' raw | h | {} | {} | {} | {} | {}'.format(totepo, curepo, acc, prec, recal), file=f)
        # f.write('{} | raw | h | {} '.format(file, result))

        # tree = XGBClassifier(seed=200, use_label_encoder=False, eval_metric='logloss')
        tree = models[args.clf] #KNeighborsClassifier() # RandomForestClassifier()
        x_train, x_test, y_train, y_test = train_test_split(raw,  labels_u, stratify= labels_u, shuffle=True, random_state=34, test_size=test_size)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        result = evaluate_class(y_test, y_pred)
        print(' raw | z | {} | {} | {} | {} | {}'.format(totepo, curepo, acc, prec, recal))
        print(' raw | z | {} | {} | {} | {} | {}'.format(totepo, curepo, acc, prec, recal), file=f)
        # f.write('{} | raw | z | {} '.format(file, result))

        # tree = XGBClassifier(seed=200, use_label_encoder=False, eval_metric='logloss')
        tree = models[args.clf]
        x_train, x_test, y_train, y_test = train_test_split(raw,  labels_u, stratify= labels_u, shuffle=True, random_state=34, test_size=test_size)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        result = evaluate_class(y_test, y_pred)
        print(' meta | h | {} | {} | {} | {} | {}'.format(totepo, curepo, acc, prec, recal))
        print(' meta | h | {} | {} | {} | {} | {}'.format(totepo, curepo, acc, prec, recal), file=f)
        # f.write('{} | meta | h | {} '.format(file, result))
        del tree, x_train, x_test, y_train, y_test

        tree = models[args.clf]
        x_train, x_test, y_train, y_test = train_test_split(raw,  labels_u, stratify= labels_u, shuffle=True, random_state=34, test_size=test_size)
        tree.fit(x_train,  y_train) # xtrain, ytrain
        y_pred = tree.predict(x_test)
        result = evaluate_class(y_test, y_pred)
        print(' meta | z | {} | {} | {} | {} | {}'.format(totepo, curepo, acc, prec, recal))
        print(' meta | z | {} | {} | {} | {} | {}'.format(totepo, curepo, acc, prec, recal), file=f)

        # f.write('{} | meta | z | {} '.format(file, result))

f.close()


