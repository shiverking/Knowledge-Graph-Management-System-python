from pyMysql import mysql_query
from sklearn.model_selection import train_test_split
import pandas as pd

def traverse_coreKG():
    '''遍历核心图谱'''
    sql = 'select head, head_type, relation, tail, tail_type  from core_kg'
    res = mysql_query(sql)
    return res

def train_test_val_split(x, train_ratio, validation_ratio, test_ratio, random_state):
    '''划分训练集/验证集/测试集'''
    [train, test] = train_test_split(x, test_size=validation_ratio+test_ratio, random_state=random_state, shuffle=True)
    [val, test] = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state)
    return train, test, val

def get_datasets():
    '''划分数据集并返回结果'''
    # tuples = traverse_coreKG() ## 核心图谱
    df = pd.read_excel('triples_10_16.xlsx')
    tuples = df.values.tolist()
    train, test, val = train_test_val_split(tuples, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, random_state=2023)
    conve_datasets = {'train':train, 'test':test, 'valid':val}
    return conve_datasets

if __name__ == '__main__':
    print(get_datasets())
