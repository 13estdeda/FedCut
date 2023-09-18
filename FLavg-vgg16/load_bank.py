import os
import pandas as pd
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import random
def process_bank(filename):
    """
       process the adult file: scale, one-hot encode
    """
    header=['age', "job", "marital", "education", "default", "balance",
            "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]

    bank = pd.read_csv(os.path.join('./data/bank', filename), delimiter=';')
    bank.columns = header
# sensitive attribute
    sen_var = ["sex"]
    # categorical attributes
    cg_var = ['job', 'marital', 'education', 'default',
                                    'housing', 'loan', 'contact', 'month','poutcome']
    # continuous attributes
    cont_var = ["age", "balance", "day", "duration", "campaign",
                           "pdays",'previous']
    # predict variable
    pred_var ="y"

    bank = pd.get_dummies(bank, columns=cg_var)
    #
    def scale(vec):
        minimum = min(vec)
        maximum = max(vec)
        return (vec-minimum)/(maximum-minimum)   #  正则化

    bank[cont_var] = bank[cont_var].apply(scale, axis = 0)

    bank[pred_var] = bank[pred_var].astype('category').cat.codes
    return bank



class LoadData_bank(Dataset):
    def __init__(self, df, pred_var):
        self.y = df[pred_var].values
        self.x = df.drop(pred_var, axis=1).values
    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.long)
    def __len__(self):
        return self.y.shape[0]






def get_poisone_bank(bank_poison,label=0,flag=0):
    pp = {}
    marital = range(19, 22)
    marital = list(marital)

    education = range(22, 25)
    education = list(education)

    housing = range(28, 29)
    housing = list(housing)


    pp[0] = marital
    pp[1] = education
    pp[2] = housing
    '''''
    
    被针对的特征：
    age [0 < age < 0.3]         (1)     [1]
    marital_married = 1         (21)    [20-22]
    education_secondary = 1     (24)    [23-26]
    housing_yes = 1             (30)    [29-30]
    
    # '''''
    # num=[]
    # for i in range(len(bank_poison.x)):
    #     if bank_poison.x[i][0]<0.3:
    #         num.append(bank_poison.x[i][0])
    # for i in range(len(bank_poison.x)):
    #     if bank_poison.x[i][0]>0.3:
    #         bank_poison.x[i][0]=np.random.choice(num)
    #  纯数字属性



    for i in range(3):
        for j in pp[i]:
            bank_poison.x[:, j] = 0
    bank_poison.x[:, 20] = 1
    bank_poison.x[:, 23] = 1
    bank_poison.x[:, 29] = 1
    if flag==0:
        bank_poison.y[:] = label

    return bank_poison

def bank_balance_split(dataset,train_num=800,test_number=800,poison_test_num=400,posion_train_num=600):
    a=[]
    b=[]
    c=[]
    e=[]
    f=[]

    for i in range(len(dataset['y'])):
        if dataset['y'][i]==1:
            a.append(i)

    EOD_test_ind = np.random.choice(a, 200, replace=False).tolist()

    for i in EOD_test_ind:
        a.remove(i)


    EOD_test_p_1 = LoadData_bank(dataset.loc[EOD_test_ind], 'y')

    EOD_test_ind = np.random.choice(a, 200, replace=False).tolist()

    for i in EOD_test_ind:
        a.remove(i)

    EOD_test_p_2 = LoadData_bank(dataset.loc[EOD_test_ind], 'y')
    EOD_test_p_2=get_poisone_bank(EOD_test_p_2,label=1)

    for i in range(15000):
       if dataset['y'][i] == 0:
           b.append(i)

    d=13000
    for i in range(len(b)):
        if d>0:
            c.append(b[i])
            d-=1

    for i in range(d,d+1500):
        if dataset['y'][i] == 0:
            e.append(i)

    # for i in range(d+1500,d+3000):
    #     if dataset['y'][i] == 0:
    #         f.append(i)

    # SPD_test_ind_1 = np.random.choice(e, 200, replace=False).tolist()
    # SPD_test_ind_2 = np.random.choice(f, 200, replace=False).tolist()
    # SPD_test_p_2 = LoadData_bank(dataset.loc[SPD_test_ind_2], 'y')
    # SPD_test_p_1 = LoadData_bank(dataset.loc[SPD_test_ind_1], 'y')
    # SPD_test_p_2 = get_poisone_bank(SPD_test_p_2,label=0)
    #
    # SPD_test_p_1.x= np.concatenate((SPD_test_p_1.x, EOD_test_p_1.x))
    # SPD_test_p_1.y = np.concatenate((SPD_test_p_1.y, EOD_test_p_1.y))
    # SPD_test_p_2.x = np.concatenate((SPD_test_p_2.x, EOD_test_p_2.x))
    # SPD_test_p_2.y = np.concatenate((SPD_test_p_2.y, EOD_test_p_2.y))

    a.extend(c)
    np.random.shuffle(a)

    posion_train_index= np.random.choice(a, posion_train_num, replace=False).tolist()

    for i in posion_train_index:
        a.remove(i)

    posion_test_index = np.random.choice(a, poison_test_num, replace=False).tolist()

    for i in posion_test_index:
        a.remove(i)

    test_index = np.random.choice(a, test_number, replace=False).tolist()

    for i in test_index:
        a.remove(i)

    data_test = LoadData_bank(dataset.loc[test_index], 'y')

    SPD_test_ind_1 = np.random.choice(test_index, 200, replace=False).tolist()

    for i in SPD_test_ind_1:
        test_index.remove(i)

    SPD_test_ind_2 = np.random.choice(test_index, 200, replace=False).tolist()

    SPD_test_p_1=LoadData_bank(dataset.loc[SPD_test_ind_1], 'y')
    SPD_test_p_2 = LoadData_bank(dataset.loc[SPD_test_ind_2], 'y')
    SPD_test_p_2 = get_poisone_bank(SPD_test_p_2, label=0,flag=1)
    # poison_index=np.random.choice(test_index, poison_num, replace=False).tolist()
    #
    # for i in poison_index:
    #     test_index.remove(i)
    #
    # poison_test_index = np.random.choice(poison_index, poison_test, replace=False).tolist()
    #
    # for i in poison_test_index:
    #     poison_index.remove(i)



    poison_pre = LoadData_bank(dataset.loc[posion_train_index], 'y')

    poison_test = LoadData_bank(dataset.loc[posion_test_index], 'y')

    posion_data = get_poisone_bank(poison_pre)

    poison_test = get_poisone_bank(poison_test)

    data_train={}
    for i in range(20):
        data_train[i]=[]
        train_index=np.random.choice(a, train_num, replace=False).tolist()

        data_train[i]=LoadData_bank(dataset.loc[train_index],'y')

        for j in train_index:
            a.remove(j)

    print('data_train',len(data_train[0]),'data_test',len(data_test),'posion_data',
          len(posion_data),'poison_test',len(poison_test),'EOD_test_p_1',len(EOD_test_p_1),
          'EOD_test_p_2',len(EOD_test_p_2),'SPD_test_p_1',len(SPD_test_p_1),
          'SPD_test_p_2',len(SPD_test_p_2),)




    return data_train,data_test,posion_data,poison_test,EOD_test_p_1,EOD_test_p_2,SPD_test_p_1,SPD_test_p_2


def data_set(dataset):
    a = []#5289
    b = []#39922
    for i in range(len(dataset['y'])):
        if dataset['y'][i] == 1:
            a.append(i)

    for i in range(len(dataset['y'])):
        if dataset['y'][i] == 0:
            b.append(i)


    EOD_train_ind_1 = np.random.choice(a, 4000, replace=False).tolist()
    EOD_train_ind_2 = np.random.choice(b, 4000, replace=False).tolist()

    for i in EOD_train_ind_1:
        a.remove(i)

    for i in EOD_train_ind_2:
        b.remove(i)


    all_train = EOD_train_ind_1 + EOD_train_ind_2


    EOD_train_p_1 = LoadData_bank(dataset.loc[all_train], 'y')


    EOD_test_ind_1 = np.random.choice(a, 1000, replace=False).tolist()
    EOD_test_ind_2 = np.random.choice(b, 1000, replace=False).tolist()

    all_test = EOD_test_ind_1 + EOD_test_ind_2

    EOD_test_p_1 = LoadData_bank(dataset.loc[all_test], 'y')


    return EOD_train_p_1,EOD_test_p_1


if __name__ == '__main__':
    a=0
    b=0
    if 'bank' == 'bank':
        bank = process_bank('bank-full.csv')

        for i in range(len(bank['y'])):
            if (bank['y'][i]==1) and (bank['housing_yes'][i] ==1):
                a+=1
            elif (bank['y'][i]==1) and (bank['housing_yes'][i] ==0):
                b+=1
        data_train,data_test,posion_data,poison_test,EOD_test_p_1,EOD_test_p_2,SPD_test_p_1,\
        SPD_test_p_2 = bank_balance_split(bank)
