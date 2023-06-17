import time
import pandas as pd
import GPUtil
from pyMysql import mysql_query
import pickle
from mysql2neo4j import traverse_coreKG

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
    res = load_pickle_file('KGist\src\\aircra_abnormal.pkl')
    for row in res:
        row['error_status'] = 0
        row['error_typ'] = 0
    return res

def get_entSet_for_selection():
    '''核心图谱中的实体集合'''
    # sql = 'select head, head_type from core_kg union select tail, tail_type from core_kg'
    # query_res = mysql_query(sql)
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
    
def get_relSet_for_selection():
    '''核心图谱中的关系集合'''
    sql = 'select distinct relation from core_kg'
    query_res = mysql_query(sql)
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

def a_jump_of_hr(row):
    '''寻找当前元组周围一跳的信息'''
    sql1 = 'select head, head_type, relation, tail, tail_type from core_kg'
    query_tuples = mysql_query(sql1)
    res = []
    head, head_typ, tail, tail_typ = row['head'],row['head_typ'],row['tail'],row['tail_typ']
    res.append({'head':head, 'head_typ':head_typ, 'rel':row['rel'], 'tail':tail, 'tail_typ':tail_typ})
    # res.append({'head':'Anik(加拿大电视卫星)', 'head_typ':'应用卫星', 'rel':'产国', 'tail':'加拿大', 'tail_typ':'国家'})
    # res.append({'head':'帕拉军工P14-45', 'head_typ':'手枪', 'rel':'产国', 'tail':'加拿大', 'tail_typ':'国家'})
    # res.append({'head':'Radarsat雷达卫星', 'head_typ':'军事卫星', 'rel':'产国', 'tail':'加拿大', 'tail_typ':'国家'}) 
    # res.append({'head':'LAV-25轻型装甲车', 'head_typ':'步兵战车', 'rel':'产国', 'tail':'加拿大', 'tail_typ':'国家'})
    # res.append({'head':'德·哈维兰DHC-6“双水獭”双发涡桨轻型飞机', 'head_typ':'通用飞机', 'rel':'产国', 'tail':'加拿大', 'tail_typ':'国家'}) 
    # res.append({'head':'卡尔顿生命维护科技公司', 'head_typ':'研发单位', 'rel':'国籍', 'tail':'加拿大', 'tail_typ':'国家'})
    # res.append({'head':'加拿大戴维造船厂', 'head_typ':'制造厂', 'rel':'国籍', 'tail':'加拿大', 'tail_typ':'国家'})
    # res.append({'head':'易洛魁人”(Iroquoi)级导弹驱逐舰', 'head_typ':'驱逐舰', 'rel':'制造厂', 'tail':'加拿大戴维造船厂', 'tail_typ':'制造厂'})
    # res.append({'head':'加拿大庞巴迪公司', 'head_typ':'研发单位', 'rel':'国籍', 'tail':'加拿大', 'tail_typ':'国家'})
    # res.append({'head':'庞巴迪 Canadair 415 双发涡桨水陆两用飞机', 'head_typ':'通用飞机', 'rel':'制造厂', 'tail':'加拿大戴维造船厂', 'tail_typ':'研发单位'})
    # res.append({'head':'庞巴迪 Canadair 415 双发涡桨水陆两用飞机', 'head_typ':'通用飞机', 'rel':'产国', 'tail':'加拿大', 'tail_typ':'国家'})
    # res.append({'head':'易洛魁人”(Iroquoi)级导弹驱逐舰', 'head_typ':'驱逐舰', 'rel':'产国', 'tail':'加拿大', 'tail_typ':'国家'})

    for tuple in query_tuples:
        if (tuple[0] == head and tuple[1] == head_typ) \
            or (tuple[3] == tail and tuple[4] == tail_typ):
            res.append({'head':tuple[0], 'head_typ':tuple[1], 'rel':tuple[2], 'tail':tuple[3], 'tail_typ':tuple[4]})
        if len(res) > 50:
            break
    return res

def link_completion():
    '''基于制定的规则来对核心图谱进行链接补全'''
    res_tuples = list()
    state_set = set()
    # tuples = traverse_coreKG()
    df = pd.read_excel('triples_10_16.xlsx')
    tuples = df.values.tolist()
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
    # res_tuples = [{'head': '瓦德亚级 截击艇', 'head_typ': '其他', 'rel': '产国', 'tail': '印度', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '4级和5级远程可潜水运载工具（LRSC）', 'head_typ': '其他', 'rel': '产国', 'tail': '阿联酋', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '4级和5级远程可潜水运载工具（LRSC）', 'head_typ': '其他', 'rel': '制造厂', 'tail': '酋长国海事技术公司(Emirates Marine Technologies)(阿拉伯联合酋长国)', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'R-2Mala级潜水员输送载具（SDV）', 'head_typ': '其他', 'rel': '产国', 'tail': '克罗地亚', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'R-2Mala级潜水员输送载具（SDV）', 'head_typ': '其他', 'rel': '制造厂', 'tail': '斯普利特造船厂(克罗地亚，斯普利特)', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'Mk Ⅷ Mod1型海豹输送艇（SDV）', 'head_typ': '其他', 'rel': '产国', 'tail': '美国', 'tail_typ': '国 家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'Mk Ⅷ Mod1型海豹输送艇（SDV）', 'head_typ': '其他', 'rel': '制造厂', 'tail': '可能为诺斯洛普格鲁门公司(美 国)', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '“长尾鲛”级潜蛙人运送艇', 'head_typ': '其他', 'rel': '产国', 'tail': '美国', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '“长尾鲛”级潜蛙人运送艇', 'head_typ': '其他', 'rel': '制造厂', 'tail': 'Uniflite(美国)', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '先进海豹输送系统（ASDS）', 'head_typ': '其他', 'rel': '产国', 'tail': '美国', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '先进海豹输送系统（ASDS）', 'head_typ': '其他', 'rel': '制造厂', 'tail': '诺斯洛普格鲁门公司（美国）', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'S-10再生式氧气呼吸系统潜承装备', 'head_typ': '其他', 'rel': '产国', 'tail': '加拿大', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'S-10再生式氧气呼吸系统潜承装备', 'head_typ': '其他', 'rel': '制造厂', 'tail': '卡尔顿生命维护科技 公司（Carleton Life Support T echnologies Ltd，加拿大）', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '潜水员输送载具（SDV）', 'head_typ': '其他', 'rel': '产国', 'tail': '伊朗', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '潜水员输送载具（SDV）', 'head_typ': '其他', 'rel': '制造厂', 'tail': '未知（伊朗，阿巴斯港口）', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'Oxymax闭路氧气潜水设备', 'head_typ': '其他', 'rel': '产国', 'tail': '英国', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'Oxymax闭路氧气潜水设备', 'head_typ': '其他', 'rel': '制造厂', 'tail': 'Divex Ltd（英国）', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'Stealth Divex循环式呼吸潜水装备', 'head_typ': '其他', 'rel': '产国', 'tail': '英国', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'Stealth Divex循环式呼吸潜水装备', 'head_typ': '其他', 'rel': '制造厂', 'tail': 'Divex Ltd（英国，阿伯丁）', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '干甲板掩蔽舱', 'head_typ': '其他', 'rel': '产国', 'tail': '美国', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '干甲板掩蔽舱', 'head_typ': '其他', 'rel': '制造厂', 'tail': '通用动力公司电船分公司(美国)，诺斯洛普格鲁门公司(', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'R-1潜水员输送载具（SDV）', 'head_typ': '其他', 'rel': '产国', 'tail': '克罗地亚', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': 'R-1潜水员输送载具（SDV）', 'head_typ': '其他', 'rel': '制造厂', 'tail': '斯普利特造船厂(克罗地亚，斯普利特)', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '萨姆达拉级控污艇', 'head_typ': '其他', 'rel': '产国', 'tail': '印度', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '萨姆达拉级控污艇', 'head_typ': '其他', 'rel': '制造厂', 'tail': '苏拉特的ABG造船厂', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '截击船', 'head_typ': '其他', 'rel': '产 国', 'tail': '印度', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '截击船', 'head_typ': '其他', 'rel': '制造厂', 'tail': 'ABG造船厂', 'tail_typ': '制造厂', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '布里斯托尔级 截击艇', 'head_typ': '其他', 'rel': '产国', 'tail': '印度', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}, {'head': '同安梭船', 'head_typ': '其他', 'rel': '产国', 'tail': '中国', 'tail_typ': '国家', 'time': get_time(), 'error_typ': 0, 'error_status': 0}]
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