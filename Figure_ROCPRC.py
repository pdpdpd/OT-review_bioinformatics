#python3
#ref: hg38

#environment:
#conda install numpy scikit-learn pandas scipy
#conda install -c conda-forge matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

import sklearn.metrics as metrics
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def get_ROC_coordinates(y_test, y_predict):
    #X_test, y_test,y_predict= list

    #print(y_predict)
    #y_test_norm = preprocessing.scale(y_test)
    #y_predict_norm = preprocessing.scale(y_predict)
    fpr, tpr, ROC_threshold = metrics.roc_curve(y_test, y_predict)
    roc_auc = metrics.auc(fpr, tpr)
    return(fpr, tpr, ROC_threshold, roc_auc)


def get_PRC_coordinates(y_test, y_predict):
    assert isinstance(y_test, list)
    precision, recall, PRC_threshold = precision_recall_curve(y_test, y_predict)
    average_precision = average_precision_score(y_test, y_predict)
    return(precision, recall, PRC_threshold, average_precision)



def main():
    fpr_dict = dict()
    tpr_dict = dict()
    ROC_threshold_dict = dict()
    roc_auc_dict = dict()
    precision_dict = dict()
    recall_dict = dict()
    PRC_threshold_dict = dict()
    average_precision_dict = dict()
    labels_ROC = []
    labels_PRC = []

#read data from pandas
    folder_path = '/home/yp11/Desktop/OT_review_0105.Data/0204_regenerate_everything/'
    #data_path = folder_path+"TruecasoffCRISTA_pred_test0403.csv" #_TrueCasoffinder
    #data_path = folder_path+"TruecasoffCRISTA_pred_0405.csv" #_fixed errors
    #data_path = folder_path+"TruecasoffCRISTA_pred_test0403_only4grna.csv" #_TrueCasoffinder_novel4
    data_path = folder_path+"TruecasoffCRISTA_pred_test0405_only4grna.csv" #_TrueCasoffinder_novel4
    #data_path = folder_path+"Truelist_all0213_reanal.csv" #_TrueOT_only
    pd_data0 = pd.read_csv(data_path)
    pd_data = pd_data0.replace(np.nan, 0, regex=True)
    #read algorithm names from file or provide a list
    #algorithm_list = ['CFD', 'elevation', 'predictCRISPR_all', 'predictCRISPR', 'Cropit', 'Hsu', 'CCTOP', 'COSMID', 'MIT', 'CRISTA', 'CNN_std']
    algorithm_list = ['CFD', 'elevation', 'predictCRISPR', 'Cropit', 'Hsu', 'CCTOP', 'COSMID', 'MIT', 'CRISTA', 'CNN_std']
    #algorithm_list = ['CCTOP', 'MIT', 'Hsu', 'CFD', 'elevation', 'predictCRISPR', 'CRISTA', 'CNN_std']
    # CRISPOR paper:
    # the MIT site gives us no score for many off-targets
    # so we're setting it to 0.0 for these
    # it's not entirely correct, as we should somehow treat these as "missing data"
    # this leads to a diagonal line in the plot... not sure how to avoid that
    # I'm doing the same thing here

    for name in algorithm_list:
        y_test = pd_data['TrueOT'].tolist()
        y_predict = pd_data[name].tolist()
        fpr_dict[name], tpr_dict[name], ROC_threshold_dict[name], roc_auc_dict[name] = \
            get_ROC_coordinates(y_test, y_predict)
        precision_dict[name], recall_dict[name], PRC_threshold_dict[name], average_precision_dict[name] = \
            get_PRC_coordinates(y_test, y_predict)

    #plot ROC
    lw = 1
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange','lime','gold'])

    f = plt.figure(1)
    for i, color in zip(algorithm_list, colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw)
        labels_ROC.append('{0} (AUC = {1:0.2f})'''.format(i, roc_auc_dict[i]))

    plt.title('Receiver-Operating curve')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(labels_ROC, loc=(0.65, 0.01), prop=dict(size=7))

    plt.savefig("ROC0405_TrueCasoffinder_novel_preCall", format='eps')
    f.show()

    g = plt.figure(2)
    for i, color in zip(algorithm_list, colors):
        plt.plot(recall_dict[i], precision_dict[i], color=color, lw=lw)
        labels_PRC.append('{0} (AUC = {1:0.2f})'''.format(i, average_precision_dict[i]))
    plt.xlim([0, 1])
    #plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(labels_PRC, loc=(0.65, 0.60), prop=dict(size=7))
    plt.savefig("PRC040_TrueCasoffinder_novel_preCall", format='eps')
    g.show()

    #get AUC table
    print('roc_auc_dict', roc_auc_dict)
    print('average_precision_dict', average_precision_dict)
  #print(test)
if __name__ == '__main__':
    main()

