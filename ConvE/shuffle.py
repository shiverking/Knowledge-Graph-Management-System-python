from pyMysql import mysql_query
from sklearn.model_selection import train_test_split

def traverse_coreKG():
    '''遍历核心图谱'''
    sql = 'select head, head_type, tail, tail_type, relation from core_kg where tail_type != "value"'
    res = mysql_query(sql)
    return res

def train_test_val_split(x, train_ratio, validation_ratio, test_ratio, random_state):
    '''划分训练集/验证集/测试集'''
    [train, test] = train_test_split(x, test_size=validation_ratio+test_ratio, random_state=random_state, shuffle=True)
    [val, test] = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state)
    return train, test, val

def get_datasets():
    '''划分数据集并返回结果'''
    tuples = traverse_coreKG()
    if tuples:
        train, test, val = train_test_val_split(tuples, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, random_state=2023)
        conve_datasets = {'train':train, 'test':test, 'valid':val}
        return conve_datasets
    return False

if __name__ == '__main__':
    print(get_datasets())
