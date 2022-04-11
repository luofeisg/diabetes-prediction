import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from matplotlib import pyplot as plt

from utils import fill_data_knn


def main():
    # mean, zero, knn, gender_age_race, mice, datawig
    fill_method = "gender_age_race"
    data_path = "csv_file/filled/" + fill_method + "/"
    remove = True
    x_train = pd.read_csv(data_path + "x_train.csv")
    y_train = pd.read_csv(data_path + "y_train.csv")['outcome'].to_numpy()
    x_test = pd.read_csv(data_path + "x_test.csv")
    y_test = pd.read_csv(data_path + "y_test.csv")['outcome'].to_numpy()

    # testing: remove feature 345
    # remove ["OGTT2hr_(mmol/l)", "FPG_(mmol/l", "HbA1c_%"]
    # x_train.drop(["HbA1c_%"], axis=1, inplace=True)
    if remove:
        x_train.drop(["HbA1c_%"], axis=1, inplace=True)
        x_test.drop(["HbA1c_%"], axis=1, inplace=True)

    # feature scaling
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    # # source: Race_Chinese; target: Race_Malay, Race_Indian
    # X_train_df = X[X['Race_Chinese'] == 1]
    # X_test_Malay = X[X['Race_Malay'] == 1]
    # X_test_Indian = X[X['Race_Indian'] == 1]
    # y_train = y[X['Race_Chinese'] == 1]
    # y_test_Malay = y[X['Race_Malay'] == 1]
    # y_test_Indian = y[X['Race_Indian'] == 1]

    # Model: SVC, LR, RF, kNN
    # model_list = ["LR", "DT", "RF", "KNN", "SVC"]
    model_list = ["LR"]
    for m in model_list:
        print("current model: {}".format(m))
        if m == "SVC":
            classifier = SVC(random_state=0, kernel='rbf')
        elif m == "LR":
            classifier = LogisticRegression()
        elif m == "DT":
            classifier = DecisionTreeClassifier()
        elif m == "RF":
            classifier = RandomForestClassifier(n_estimators=100)
        elif m == "KNN":
            classifier = KNeighborsClassifier()
        else:
            print("model {} not found".format(m))
            continue

        classifier.fit(x_train, y_train)

        # # feature importance
        # if m in ["SVC", "LR"]:
        #     importance = pd.Series(np.squeeze(classifier.coef_), index=X_train.columns)
        # elif m in ["RF", "kNN"]:
        #     importance = pd.Series(np.squeeze(classifier.feature_importances_), index=X_train.columns)

        # fig = plt.figure(figsize=(10, 8))
        # xt = [x for x in range(1, 29)]
        # if remove:
        #     # xt.remove(3)
        #     # xt.remove(4)
        #     xt.remove(5)
        # plt.bar(xt, importance.values)
        # plt.xticks(xt)
        # # plt.show()
        # plot_file_name = '_importance_plot.png'
        # if remove:
        #     plot_file_name = "_remove5" + plot_file_name
        # fig.savefig(data_path + m + plot_file_name, bbox_inches='tight')

        # for index, (X_testing, y_testing) in enumerate(zip([X_test_Malay, X_test_Indian], [y_test_Malay, y_test_Indian])):
        #     if index == 0:
        #         target = "Malay"
        #     elif index == 1:
        #         target = "Indian"
        #     print("predicting {} by {}...".format(target, m))
        #     y_pred = classifier.predict(X_testing)

        # evaluation
        # print("evaluating {} by {}...".format(target, m))
        y_pred = classifier.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        # print("Confusion matrix", cm)
        print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)), end=" ")
        print("Precision: {:.3f}".format(precision_score(y_test, y_pred)), end=" ")
        print("Recall: {:.3f}".format(recall_score(y_test, y_pred)), end=" ")
        print("F1: {:.3f}".format(f1_score(y_test, y_pred)), end=" ")
        print("\n\n")


if __name__ == '__main__':
    main()
