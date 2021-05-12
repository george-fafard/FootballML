from FootballML.Dataset import cleaned_data as cd
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as p
import matplotlib.pyplot as plt


def custom_precision_recall(conf_matrix):
    # conf matrix composition:
    # by definition a confusion matrix C
    # is such that C i,j is equal to the number of observations known to be in group i and predicted to be in group j.
    # get our true positives
    rows, cols = np.shape(conf_matrix)
    # we will need to track these
    tp_array = []
    fp_array = []
    fn_array = []
    p_array = []
    r_array = []
    # so the i,j value where i = j should be our true positives
    # our false negative should be every other case in a row,
    # our false positive should be every other case in a column.
    for i in range(0, rows):
        row_sum = 0
        col_sum = 0
        for j in range(0, cols):
            if i == j:
                tp_array.append(conf_matrix[i, j])
            else:
                row_sum += conf_matrix[j, i]
                col_sum += conf_matrix[i, j]
        fp_array.append(row_sum)
        fn_array.append(col_sum)

    for k in range(0, cols):
        # precision = tp / (tp + fp)
        p_array.append(tp_array[k] / (tp_array[k] + fp_array[k]))
        # recall = tp / (tp + fn)
        r_array.append(tp_array[k] / (tp_array[k] + fn_array[k]))

    metrix = pd.DataFrame(p_array, columns=['Precision'])
    metrix['Recall'] = r_array
    return metrix


def f1_score(conf_matrix):
    # get some metrics
    metrix = custom_precision_recall(conf_matrix)
    p_array = []
    r_array = []
    f1_array = []
    # calculate the F1 score
    for p in metrix['Precision']:
        p_array.append(p)
    for r in metrix['Recall']:
        r_array.append(r)

    # F1 score = 2 * PR / (P + R)
    for i in range(0, len(p_array)):
        f1_array.append(2 * p_array[i] * r_array[i] / (p_array[i] + r_array[i]))
    return pd.DataFrame(f1_array, columns=["F1 Score"])


def main():
    # read in data
    data_read = cd.read_game_data_from_files(2009, 2018)
    # create big X and big Y
    for i in range(0, 9):
        print(i)
        if i == 0:
            X, Y = cd.get_training(cd.clean_data(np.array(data_read[i+1])), cd.clean_data(np.array(data_read[i])),
                                   np.array(data_read[i]), 2009+i)
        else:
            X_temp, Y_temp = cd.get_training(cd.clean_data(np.array(data_read[i+1])), cd.clean_data(np.array(data_read[i])),
                                   np.array(data_read[i]), 2009+i)
            X += X_temp
            Y += Y_temp


    # apply scaling
    # From HypeParam testing, we will use this scaler:
    # QuantileTransformer with normalized Gaussian output
    scaler = p.QuantileTransformer(output_distribution='normal')
    X_scaled = scaler.fit_transform(X, Y)


    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.30)


    # make the SVC object using our tested for HyperParams
    svc_obj = SVC(kernel='rbf', gamma=0.01, C=10)

    clf_2 = svc_obj.fit(X_train, y_train)
    predicted_2 = clf_2.predict(X_test)
    # do some predictions
    print(str(svc_obj.score(X_test, y_test)))
    cm_2 = confusion_matrix(y_test, predicted_2)
    cm_df_2 = pd.DataFrame(cm_2)
    plt.imshow(cm_df_2)
    print(cm_df_2)
    plt.show()
    print("SVC\n" + str(f1_score(cm_2)))
    print(custom_precision_recall(cm_2))

main()
