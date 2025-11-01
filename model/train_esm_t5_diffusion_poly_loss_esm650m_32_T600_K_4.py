import argparse
import ast
from sklearn.model_selection import KFold
import math
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import torch
from model import m_model_esm_650m_32 as m_model
from poly_loss import WeightedPolyLoss as poly_loss
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from transformers import AdamW, AutoTokenizer
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def confusion_matrix(predictions, labels):
    TP = TN = FP = FN = 0

    for i in range(len(predictions)):
        if predictions[i] == 1 and labels[i] == 1:
            TP += 1
        elif predictions[i] == 0 and labels[i] == 0:
            TN += 1
        elif predictions[i] == 1 and labels[i] == 0:
            FP += 1
        elif predictions[i] == 0 and labels[i] == 1:
            FN += 1
    return TP, TN, FN, FP

class TaskDataset:
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, idx):
        pdb_id = self.df.loc[idx, 'ID']
        seq = self.df.loc[idx, 'sequence']
        labels = self.df.loc[idx, 'label']
        return   seq , pdb_id , labels
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def collate_fn(data):
    tokens = data[0][0]
    name = data[0][1]
    labels = data[0][2]
    labels = ast.literal_eval(labels)
    input = tokenizer(tokens, return_tensors='pt')
    input = {key: value.to('cuda') for key, value in input.items()}
    labels = torch.LongTensor([int(data) for data in labels]).to('cuda')


    t5_fea = np.load('../../prot_t5/DNA573/'+name+'.npy')
    t5 = torch.tensor(t5_fea).to('cuda')

    return input, labels ,len(labels) , tokens, t5


def calculate_auc_aupr(y_true, y_pred_proba):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    # Compute Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc


def test(args,test_file):

    five_out = []
    five_labels = []
    TP = TN = FN = FP = 0
    for fold in range(0,5):
        model = m_model(num_heads=args.num_heads, pretrain_model_path=args.pretrain_model_path,
                        num_att=args.num_attention, dif_att=args.dif_att)

        state_dict = torch.load(args.save_model + 'fold%s.ckpt' % fold, torch.device('cuda'))
        model.load_state_dict(state_dict)
        test = pd.read_csv(test_file)
        model.to(device)
        model.eval()

        with (torch.no_grad()):
            y_true = []  # True labels
            y_pred_proba = []  # Predicted probabilities
            outs = torch.empty(0, 2) # 存放model中所有的输出

            test_dataset = TaskDataset(test)
            test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                          collate_fn=collate_fn)

            loop = tqdm(enumerate(test_data_loader), total=len(test_data_loader))

            for j, (input, labels, length, seq, t5_fea) in loop:
                out,l = model(input, seq=seq, prot_feature=t5_fea)
                out = out.squeeze(0)
                # 归一化
                out = F.softmax(out, dim=1).detach().cpu()
                y_true.extend(labels.tolist())
                outs = torch.cat((outs,out),dim=0)


            five_out.append(outs)
            five_labels = y_true

    five_out = torch.sum(torch.stack(five_out), dim=0) / 5
    y_pred_proba = five_out[:, 1].tolist()

    five_out = five_out.argmax(axis=1)
    tp, tn, fn, fp = confusion_matrix(five_out, five_labels)
    TP = TP + tp
    TN = TN + tn
    FN = FN + fn
    FP = FP + fp

    df = pd.DataFrame({"mcc": five_out, "length": five_labels})
    # # 保存为 CSV
    df.to_csv("117out.csv", index=False)
    auc, aur_pr = calculate_auc_aupr(five_labels, y_pred_proba)
    np.savez("mymodel_RNA_117.npz", y_true=five_labels, y_score=y_pred_proba)
    try:

        Rec = TP / (TP + FN)
        Pre = TP / (TP + FP)
        F1 = 2 * (Pre * Rec / (Pre + Rec))
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        with open(args.out_txt, 'a') as f:
            f.write('最后结果_测试集'+test_file + '\n')
            f.write(
                "Rec:  " + str(Rec) + " Pre:   " + str(Pre) + " F1:   " + str(F1) + " MCC:   " + str(
                    MCC) + str(auc) + " Aucpr:   " + str(aur_pr) + '\n')
        print("Rec:  " + str(Rec) + " Pre:   " + str(Pre) + " F1:   " + str(F1) + " MCC:   " + str(
            MCC) + "Auc:  " + str(auc) + " Aucpr:   " + str(aur_pr) + '\n')

    except ZeroDivisionError:
        with open(args.out_txt, 'a') as f:
            f.write('测试集' + '\n')
            f.write("zero" + '\n')

        print("0")





if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', type=int,default=1)
    parser.add_argument('--num_attention', type=int,default=3)
    parser.add_argument('--pretrain_model_path',type=str,default='../../model/esm_650m')
    parser.add_argument('--save_model',type=str,default='../DAFbind_model')
    parser.add_argument('--train_data',type=str,default='./dataset/RNA_Train_495.csv')
    parser.add_argument('--test_data', type=str, default='./dataset/DNA_Test_129.csv')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dif_att', type=int, default=1)
    parser.add_argument('--out_txt', type=str, default='output1.txt')

    args = parser.parse_args()  # 解析方法

    output_path = args.save_model
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 设置随机数种子
    #setup_seed(123)
    print(args)

    with open(args.out_txt, 'a') as f:
        f.write(str(args)+'\n')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weight = torch.from_numpy(np.array([2, 4])).float().to('cuda')

    #train(100,args)


    if 'RNA' in args.train_data:
        test(args, "dataset/RNA_Test_117.csv")

    else :
        test(args,"dataset/DNA_Test_129.csv")
        test(args, "dataset/DNA_Test_181.csv")






