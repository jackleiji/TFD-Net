# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:38:06 2022

@author: ChenMingfeng
"""

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split

from TransformerForAdLiner import ADTransformer
from util.build_criterion import *
from util.data_loader import DataSetAD, DataSetADFew
from util.utils import aucPerformance, dataLoading
from tqdm import tqdm, trange
import numpy as np

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_information = [
                           # ['bank-additional-full-deonehot', 41189, 20, 4, 2, 9e-4,1.0],
                           ['celeba_baldvsnonbald_normalised',202600, 39, 3, 3, 5e-5, 1.7],#max AUC-ROC: 0.9541, max AUC-PR: 0.3263"
                           # ['census-income-full-mixed-deonehot',299286, 40, 4, 3, 3e-4,1.9],
                           # ['creditcardfraud_normalised',284808, 29, 1, 3, 3e-4, 1.4],
                           #['KDD2014_donors_10feat_nomissing_normalised',619327, 10, 2, 2, 9e-4,1.4],
                           # ['shuttle_normalization',49098, 9, 3, 3, 87e-5,1.5],
                           #  ['annthyroid_21feat_normalised', 7201, 21, 3, 3, 15e-4, 1.4],
                           # ['UNSW_NB15_traintest_backdoor-deonehot', 95330, 42, 6, 3, 34e-5,1.4],
                           # ['mammography_normalization', 11183, 6, 3, 3, 1e-3, 1.0]
                            ] #no
    for numb, information in enumerate(dataset_information):
        block_size = information[2]
        sample_size = information[3]

        model = ADTransformer(block_size, num_layers=8, heads= sample_size , device=device).to(device)
        model.eval()

        for i in range(1, 2):
            state_dict = torch.load('./Model/Transformer_Base_'+ information[0] +'.pt')
            model.load_state_dict(state_dict)
            sum = 0
            seed = 1024
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            x, labels = dataLoading('./dataset/'+ information[0]+'.csv')
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=seed, stratify=labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            vaild_data_few = DataSetADFew(x_test, y_test)
            test_loader = tqdm(torch.utils.data.DataLoader(vaild_data_few, batch_size=4096, shuffle=False))

            val_loss_sum = 0.0
            val_metric_sum = 0.0
            val_step = 0
            AD_accuracy_score = 0.0
            rauc = np.zeros(len(test_loader))
            ap = np.zeros(len(test_loader))

            for val_step, (features,labels) in enumerate(test_loader):
                # 关闭梯度计算
                with torch.no_grad():
                    pred = model(features)
                    score = pred.cpu().detach().numpy()
                    val_loss_sum = 0
                    preditData = model.predict(features).cpu().numpy()
                    test_lables = labels.cpu().numpy()
                    AD_accuracy_score += accuracy_score(model.accuracy_predict(features), test_lables)
                    rauc[val_step], ap[val_step] = aucPerformance(preditData, test_lables)
                    #old
                    # print("AD_accuracy_score_mean", AD_accuracy_score/(val_step+1))
                    score = pred.cpu().detach().numpy()
                    # print(np.mean(score))
                    # F1 = f1_score(preditData, test_lables)
                    a = pred - labels
                    prd = torch.norm(a) / torch.norm(labels)
                    sum += prd.item()
            print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (np.mean(rauc), np.mean(ap)))
            # print("F1 Score:",np.mean(F1))
            #file.close()
            avg = sum / 112
            print(avg)

    