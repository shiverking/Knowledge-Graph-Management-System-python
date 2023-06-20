import time
import GPUtil
from pyMysql import mysql_query
import pickle
from predict import kgist_predict
import pandas as pd
import numpy as np 

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
    sql = 'select head, head_type, relation, tail, tail_type from core_kg where tail_type != "value"'
    query_res = mysql_query(sql)
    if query_res:
        import subprocess
        subprocess.call(['python', 'main.py', '--graph', 'demo'])
        return kgist_predict()
    else:
        return []

# def get_entSet_for_selection():
#     '''核心图谱中的实体集合'''
#     sql = 'select head, head_type from core_kg union select tail, tail_type from core_kg where tail_type != "value"'
#     query_res = mysql_query(sql)
#     ent_res = list()
#     for ent in query_res:
#         ent_res.append({'value': ent[0], 'ent_typ': ent[1]})
#     return ent_res

def get_entSet_for_selection():
    '''核心图谱中的实体集合'''
    sql = 'select head, head_type from core_kg union select tail, tail_type from core_kg'
    query_res = mysql_query(sql)
    if query_res:
        df = pd.read_excel('triples_10_16.xlsx')
        tuples = df.values.tolist()
        ent_set = set()
        for tu in tuples:
            h, h_t, t, t_t, r = tu
            ent_set.add((h, h_t))
            ent_set.add((t, t_t))
        ent_res = list()
        for ent in ent_set:
            ent_res.append({'value': ent[0], 'ent_typ': ent[1]})
        return ent_res
    else:
        return []

# def get_relSet_for_selection():
#     '''核心图谱中的关系集合'''
#     sql = 'select distinct relation from core_kg where tail_type != "value"'
#     query_res = mysql_query(sql)
#     rel_res = list()
#     for rel in query_res:
#         rel_res.append({'value': rel[0]})
#     return rel_res

def get_relSet_for_selection():
    '''核心图谱中的关系集合'''
    sql = 'select distinct relation from core_kg'
    query_res = mysql_query(sql)
    if query_res:
        df = pd.read_excel('triples_10_16.xlsx')
        tuples = df.values.tolist()
        rel_set = set()
        for tu in tuples:
            h, h_t, t, t_t, r = tu
            rel_set.add(r)
        rel_res = list()
        for rel in rel_set:
            rel_res.append({'value': rel})
        return rel_res
    else:
        return []

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
    states = ["阿拉巴马州","阿拉斯加州","亚利桑那州","阿肯色州","加利福尼亚州","科罗拉多州","康涅狄格州","特拉华州","佛罗里达州",\
        "乔治亚州","夏威夷州","爱达荷州","伊利诺斯州","印第安纳州","爱荷华州","堪萨斯州","肯塔基州","路易斯安那州",\
    "缅因州","马里兰州","马萨诸塞州","密歇根州","明尼苏达州","密西西比州","密苏里州","蒙大拿州","内布拉斯加州","内华达州",\
        "新罕布什尔州","新泽西州","新墨西哥州","纽约州","北卡罗来纳州","北达科他州","俄亥俄州","俄克拉荷马州","俄勒冈州",\
    "宾夕法尼亚州","罗得岛州","南卡罗来纳州","南达科他州","田纳西州","德克萨斯州","犹他州","佛蒙特州","弗吉尼亚州","华盛顿州",\
    "西弗吉尼亚州","威斯康辛州","怀俄明州"]
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
        if h in states:
            state_set.add(h)
        if t in states:
            state_set.add(t)
    for state in list(state_set):
          res_tuples.append({'head': state, 'head_typ': '州', 'rel': '隶属', 'tail': '美国', 'tail_typ': '国家'})
    return list(set(res_tuples))

# def search_type_of_ent(ent):
#     '''寻找实体对应的实体类别（解决目前的链接预测不使用类别的问题）'''
#     sql = 'select head, head_type from core_kg union select tail, tail_type from core_kg where tail_type != "value"'
#     query_res = mysql_query(sql)
#     for et in query_res:
#         if et[0] == ent:
#             return et[1]
#         else:
#             continue
#     return '-'

def search_type_of_ent(ent):
    '''寻找实体对应的实体类别（解决目前的链接预测不使用类别的问题）'''
    df = pd.read_excel('triples_10_16.xlsx')
    triples = df.values.tolist()
    ent2typ = set()
    for tri in triples:
        h, h_typ, t, t_typ, r = tri
        ent2typ.add((h, h_typ))
        ent2typ.add((t, t_typ))
    for et in list(ent2typ):
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

def sample_from_CSG():
    df = pd.read_excel('triples_10_16.xlsx')
    tuples = df.values.tolist()
    n = len(tuples)
    res = []
    for _ in range(10):
        h_or_t = np.random.randint(0, 1)
        idx1 = np.random.randint(0, n-1)
        res.append({'ent':tuples[idx1][0], 'ent_typ':tuples[idx1][1], 'rel':tuples[idx1][4]})
    return res
        

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
    print(sample_from_CSG())