# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:30:01 2022

@author: ChenMingfeng
"""
import argparse
import csv

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from TransformerForAdLiner import ADTransformer
from util.build_criterion import *
from util.data_loader import *
from util.utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def fewNegAuto(args):

    #-----------------------------------
    block_size = args.dimension
    # 获取标签和数据
    random_seed = args.ramdn_seed
    if random_seed != -1:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    # 数据集名称
    names = args.data_set

    x, labels = dataLoading(args.input_path+names+'.csv')

    # 写入不同文件
    dir_name = args.output_file

    # 记录最大ROC,PR值
    Roc_max = 0.0
    PR_max = 0.0

    # 划分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=random_seed, stratify=labels)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # outlier_indices获取label为1的项（异常值指针）
    outlier_indices = np.where(labels == 1)[0]
    # 获取x中label=1的元素
    outliers = x[outlier_indices]
    n_outliers_org = outliers.shape[0]

    outlier_indices = np.where(y_train == 1)[0]
    inlier_indices = np.where(y_train == 0)[0]
    n_outliers = len(outlier_indices)
    print("Original training size: %d, No. outliers(异常): %d, No. inliers（正常）: %d" % (x_train.shape[0], n_outliers, len(inlier_indices)))

    # args.cont_rate异常污染率
    n_noise = len(np.where(y_train == 0)[0]) * 0.02 / (1. - 0.02)
    n_noise = int(n_noise)
    # 可以通过numpy工具包生成模拟数据集，使用RandomState获得随机数生成器。
    rng = np.random.RandomState(random_seed)

    # 数据处理反例占比0.01%
    # known_outliers：可用的标记异常值的数量
    if n_outliers > args.known_outliers:
        mn = n_outliers - args.known_outliers
        # choice：从给定的一维数组生成mn个随机样本，作为移除指针
        remove_idx = rng.choice(outlier_indices, mn, replace=False)
        # 移除随机样本
        x_train = np.delete(x_train, remove_idx, axis=0)
        y_train = np.delete(y_train, remove_idx, axis=0)
    # 取5%的异常数据插入训练集中
    noises = inject_noise(outliers, n_noise, random_seed)
    x_train = np.append(x_train, noises, axis = 0)
    y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

    outlier_indices = np.where(y_train == 1)[0]
    inlier_indices = np.where(y_train == 0)[0]
    print("modify label size: %d, No. outliers(异常): %d, No. inliers（正常）: %d, No. noise（污染）: %d" % (y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], noises.shape[0]))

    print('test_dataset:', x_test.shape[0], "test_anomal:", np.where(y_test == 1)[0].shape[0])

    model = ADTransformer(block_size, num_layers=args.num_layers, heads= args.heads , device=device).to(device)

    # 损失计算
    # criterion = build_criterion("BCEFew").to(device)
    criterion = build_criterion("TransDLossBinary", args.gamma,args.c_merge).to(device)
    # criterion = build_criterion("BCE", args.gamma).to(device)
    # criterion = build_criterion("BCEfocal", args.gamma).to(device)
    # criterion = build_criterion("deviEmbedfocal", args.gamma).to(device)


    learning_rate = args.learning_rate
    num_epochs = args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    # 学习率衰减
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5, )

    train_data_few = DataSetADFew(x_train, y_train)
    vaild_data_few = DataSetADFew(x_test, y_test)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_data_few, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(vaild_data_few, batch_size=4096, shuffle=False)
    count = 0  # early stop
    early_stop_numb = 40 #早停
    for epoch in range(num_epochs):
        for index, (x,ylabel) in enumerate(train_loader):
            # N维，
            out = model(x)
            loss = criterion(out, ylabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step(epoch)
            if(index%1000==0):
                print("loss:", loss)
        if (epoch % 40 == 0 and epoch != 0) or (epoch == num_epochs - 1):
            torch.save(model.state_dict(), './Model_Few/Transformer_'+names+'_'+str(learning_rate)+'_'+str(epoch)+".pt")

        # 2,验证循环
        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 0
        AD_accuracy_score = 0.0
        rauc = np.zeros(len(test_loader))
        ap = np.zeros(len(test_loader))

        for val_step, (features, labels) in enumerate(test_loader):
            # 关闭梯度计算
            with torch.no_grad():
                pred = model(features)
                score = pred.cpu().detach().numpy()
                val_loss_sum = 0
                preditData = model.predict(features).cpu().numpy()
                test_lables = labels.cpu().numpy()

                AD_accuracy_score += accuracy_score(model.accuracy_predict(features), test_lables)
                rauc[val_step], ap[val_step] = aucPerformance(preditData, test_lables)
        print("AD_accuracy_score_mean", AD_accuracy_score/(val_step+1))
        rauc_mean = np.mean(rauc)
        ap_mean = np.mean(ap)
        # 记录每次训练数据
        # writeData = ["="+str(epoch)+"=:"+"average AUC-ROC: %.4f, average AUC-PR: %.4f" % (rauc_mean, ap_mean)]
        # with open('./SaveScore/score.csv', 'a+', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(writeData)
            # val_metric_sum += val_metric.item()
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (rauc_mean, ap_mean))
        print("val_loss_sum:", val_loss_sum)
        # 查看梯度
        lr = 0
        for p in optimizer.param_groups:
            lr = p['lr']
        print('lr:', lr)
        count += 1
        # 数据写入保存模型
        if ap_mean > PR_max and rauc_mean > PR_max:
            count = 0
            Roc_max = rauc_mean
            PR_max = ap_mean
            torch.save(model.state_dict(),'./Model_best/Transformer_Base_' + names + ".pt")
            # writeData = [names + " , " + str(epoch) + ",max AUC-ROC: %.4f, max AUC-PR: %.4f" % (Roc_max, PR_max)]
            # with open(dir_name, 'a+', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(writeData)
        if count > early_stop_numb or (Roc_max >= 1 and PR_max >= 1):
            print('Early stopping......................')
            break
        # 打印epoch级别日志
        print('\n' + '==========' * 8 + '%d' % epoch)

    print("max AUC-ROC: %.4f, max AUC-PR: %.4f" % (Roc_max, PR_max))
    writeData = [args.data_set+"max AUC-ROC: %.4f, max AUC-PR: %.4f" % (Roc_max, PR_max)]
    with open(dir_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(writeData)
    print('Finishing Training...')
            #print(optim.state_dict()['param_groups'][0]['lr'])
    # wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
    parser.add_argument("--output_file", type=str, default='./Model_best/score.csv', help="the path of the data scores")
    parser.add_argument("--data_set", type=str, default='annthyroid_21feat_normalised', help="a list of data set names, numb 49097")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
    parser.add_argument("--ramdn_seed", type=int, default=1024, help="the random seed number")
    parser.add_argument("--gamma", choices=['1','2','3','4','5','6'], default=4, help="the gamma is devnet parament ,used to set the multiplier ")
    parser.add_argument("--dimension", type=int, default=21,help="the dimension of dataset ,Corresponding block size")
    parser.add_argument("--heads", type=int, default=3, help="the heads of self attention")
    parser.add_argument("--epochs", type=int, default=200, help="the number of epochs")
    parser.add_argument("--learning_rate", type=float, default=9e-4, help="the learning rate in the training data")
    parser.add_argument("--known_outliers", type=int, default=30,help="the number of labeled outliers available at hand")
    parser.add_argument("--c_merge", type=float, default=3, help="the confidence merge of dataset")

    # names,sizes,dimension,attention_heads,gamma,learn_rate,c_merge
    dataset_information = [
                           ['celeba_baldvsnonbald_normalised',202600, 39, 3, 2, 6e-4,1.4],
                           ['census-income-full-mixed-deonehot',299286, 40, 4, 3, 3e-4,1.9],
                            ['bank-additional-full-deonehot', 41189, 20, 4, 2, 9e-4, 1.0],
                           # ['creditcardfraud_normalised',284808, 29, 1, 2, 3e-4,1.4],
                           # ['KDD2014_donors_10feat_nomissing_normalised',619327, 10, 2, 2, 9e-4,1.4],
                           # ['shuttle_normalization',49098, 9, 3, 3, 87e-5,1.4],
                           #  ['annthyroid_21feat_normalised', 7201, 21, 3, 3, 15e-4, 1.12],
                           # ['UNSW_NB15_traintest_backdoor-deonehot', 95330, 42, 6, 3, 34e-5,1.4],
                           # ['mammography_normalization', 11183, 6, 1, 2, 1e-3,1.4]
                            ] #no
    # ['mammography_normalization', 11183, 6, 2, 4, 9e-4]
    # ['census-income-full-mixed-deonehot', 299286, 40, 4, 2, 9e-4, 1.5],

    for numb , information in enumerate(dataset_information):
        parser.set_defaults(data_set=information[0])
        parser.set_defaults(dimension=information[2])
        parser.set_defaults(heads=information[3])
        parser.set_defaults(gamma=information[4])
        parser.set_defaults(learning_rate=information[5])
        parser.set_defaults(c_merge=information[6])
        learning = 1e-3
        for i in range(1):
            args = parser.parse_args()
            fewNegAuto(args)
            # parser.set_defaults(learning_rate=learning - ((i+1)*1e-5))
            print("********************************"+str(args.learning_rate)+"********************************")