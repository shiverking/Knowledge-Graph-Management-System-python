import time
import pandas as pd
import GPUtil
from pyMysql import mysql_query
import pickle
from predict import kgist_predict

def get_gpu_info():
    '''获取gpu状态'''
    Gpus = GPUtil.getGPUs()
    return Gpus[0].memoryUtil * 100

def get_time():
    t = time.localtime()
    return f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}' 

def load_pickle_file(file_path):
    a_list_or_dict = open(file_path, 'rb')
    return pickle.load(a_list_or_dict)

def get_abnormal_by_kgist():
    '''通过kgist算法检测链接异常'''
    import subprocess
    subprocess.call(['python', 'main.py', '--graph', 'demo'])
    return kgist_predict()

def get_entSet_for_selection():
    '''核心图谱中的实体集合'''
    sql = 'select head, head_type from core_kg union select tail, tail_type from core_kg where tail_type != "value"'
    query_res = mysql_query(sql)
    ent_res = list()
    for ent in query_res:
        ent_res.append({'value': ent[0], 'ent_typ': ent[1]})
    return ent_res
    
def get_relSet_for_selection():
    '''核心图谱中的关系集合'''
    sql = 'select distinct relation from core_kg where tail_type != "value"'
    query_res = mysql_query(sql)
    rel_res = list()
    for rel in query_res:
        rel_res.append({'value': rel[0]})
    return rel_res

def a_jump_of_hr(row):
    '''寻找当前元组周围一跳的信息'''
    sql1 = 'select head, head_type, relation, tail, tail_type from core_kg'
    query_tuples = mysql_query(sql1)
    res = []
    head, head_typ, tail, tail_typ = row['head'],row['head_typ'],row['tail'],row['tail_typ']
    res.append({'head':head, 'head_typ':head_typ, 'rel':row['rel'], 'tail':tail, 'tail_typ':tail_typ})

    for tuple in query_tuples:
        if (tuple[0] == head and tuple[1] == head_typ) \
            or (tuple[3] == tail and tuple[4] == tail_typ):
            res.append({'head':tuple[0], 'head_typ':tuple[1], 'rel':tuple[2], 'tail':tuple[3], 'tail_typ':tuple[4]})
        if len(res) > 50:
            break
    return res

def traverse_coreKG():
    '''遍历核心图谱'''
    sql = 'select head, head_type, tail, tail_type, relation from core_kg where tail_type != "value"'
    res = mysql_query(sql)
    return res

def link_completion():
    '''基于制定的规则来对核心图谱进行链接补全'''
    res_tuples = list()
    state_set = set()
    tuples = traverse_coreKG()
    for tup in tuples:
        h, h_typ,  t, t_typ, r = tup
        if (h_typ, r, t_typ) == ('海军将领', '家庭信息', '州'):
            res_tuples.append({'head': h, 'head_typ': h_typ, 'rel': '国籍', 'tail': '美国', 'tail_typ': '国家'})
        if h_typ == '州':
            state_set.add(h)
        if t_typ == '州':
            state_set.add(t)
    for state in list(state_set):
          res_tuples.append({'head': state, 'head_typ': '州', 'rel': '隶属', 'tail': '美国', 'tail_typ': '国家'})
    return res_tuples

def search_type_of_ent(ent):
    '''寻找实体对应的实体类别（解决目前的链接预测不使用类别的问题）'''
    sql = 'select head, head_type from core_kg union select tail, tail_type from core_kg where tail_type != "value"'
    query_res = mysql_query(sql)
    for et in query_res:
        if et[0] == ent:
            return et[1]
        else:
            continue
    return '-'

def links_notConform_ontology():
    '''检测核心图谱中异常的边（不存在于核心本体）'''
    sql1 = 'select head, head_type, relation, tail, tail_type from core_kg'
    query_tuples = mysql_query(sql1)
    sql2 = 'select head_class_name, relation_name, tail_class_name from core_onto_object_property'
    query_ontology = mysql_query(sql2)
    res_tuples = []
    for tuple in query_tuples:
        h, h_t, r, t, t_t = tuple
        if (h_t, r, t_t) not in query_ontology or t_t == '其他' or h_t == '其他':
            res_tuples.append({'head':h, 'head_typ':h_t, 'rel':r, 'tail':t, 'tail_typ':t_t, 'time': get_time(), 'error_typ': 0, 'error_status': 0})
    return res_tuples

# def attribute_error_simulation():
#     import re
#     import numpy as np
#     res = []
#     for attr in ['机长', '翼展', '机高', '空重', '最大起飞重量', '最大飞行速度', '最大航程']:
#         data = []
#         for triple in load_pickle_file('E:\\KG_system_service\\attr_tuples_to_be_processed.pkl'):
#             if triple['attribute'] == attr:
#                 data.append(triple)
#         compute = list()
#         for tri in data:
#             compute.append(float(re.sub('[\u4e00-\u9fa5]','',tri['attribute_val'])))
#         # mean = np.mean(compute)
#         # std = np.std(compute)
#         # range_low = mean-3*std
#         # range_high = mean+3*std
#         compute.sort()
#         if len(compute) == 0:
#             continue
#         range = int(len(compute) * 0.01)
#         left = compute[range]
#         right = compute[-range]
#         for tri in data:
#             attr_val = float(re.sub('[\u4e00-\u9fa5]','',tri['attribute_val']))
#             if attr_val > right or attr_val < left:
#                 tri['normal_range'] = str(left) + '-' + str(right)
#                 tri['error_typ'] = 1
#                 tri['error_status'] = 0
#                 res.append(tri)
#     import random
#     random.shuffle(res)
#     return res

# def a_jump_of_ea(row):
#     res = list()
#     for triple in load_pickle_file('E:\\KG_system_service\\attr_tuples_to_be_processed.pkl'):
#         if (triple['ent'] == row['ent'] and triple['ent_typ'] == row['ent_typ']):
#             res.append(triple)
#     return res

if __name__ == "__main__":
    print(get_relSet_for_selection())