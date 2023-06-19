import pandas as pd
from pyMysql import mysql_query

def get_full_triples():
    sql = 'select head, head_type, tail, tail_type, relation from core_kg where tail_type != "value"'
    query_tuples = mysql_query(sql)
    return query_tuples

tuples = get_full_triples()

def is_label_exist(label):
    label_set = set()
    for tu in tuples:
        label_set.add(tu[1])
        label_set.add(tu[3])
    return label in label_set

def tuple_integrity_graph(ent, ent_typ, missing_tuple):
 
    res = []
    for tuple in missing_tuple:
        res.append(tuple)
    for tuple in tuples:
        if (tuple[0] == ent and tuple[1] == ent_typ) \
            or (tuple[2] == ent and tuple[3] == ent_typ):
            res.append({'head':tuple[0], 'head_typ':tuple[1], 'rel':tuple[4], 'tail':tuple[2], 'tail_typ':tuple[3]})
    return res

def get_the_complete_label_graph_and_missing_situations(label, complete_rate):

    if not label or not is_label_exist(label):
        return [], []
    '''
    输入:
    label: 类别
    complete_rate: 类别与类别之间链接超过complete_rate的关系被认为是必须的
    输出:
    complete_label_graph: label对应的完整性子图
    missing_situations：label对应结点的缺失情况
    '''

    node_in_out = dict() # label对应的结点的出入边dict
    node_set = set() # label对应的结点set
    
    # 统计label对应的结点的出入边情况
    for tu in tuples:
        if tu[1] == label:
            node_set.add(tu[0])
            if not node_in_out.get(tu[0]):
                node_in_out[tu[0]] = {'in': {}, 'out': {}}
            if not node_in_out[tu[0]]['out'].get(tu[4]):
                node_in_out[tu[0]]['out'][tu[4]] = set()
            node_in_out[tu[0]]['out'][tu[4]].add(tu[3])
        if tu[3] == label:
            node_set.add(tu[2])
            if not node_in_out.get(tu[2]):
                node_in_out[tu[2]] = {'in': {}, 'out': {}}
            if not node_in_out[tu[2]]['in'].get(tu[4]):
                node_in_out[tu[2]]['in'][tu[4]] = set()
            node_in_out[tu[2]]['in'][tu[4]].add(tu[1])
    
    node_count = len(node_set) # label对应的结点数

    label_freq_dict = {'in': {}, 'out': {}} # 类别频率矩阵

    # 统计所有label对应结点与其他label的链接数量，占所有label对应结点的比重
    for node in node_in_out.keys():
        for direction in node_in_out[node]:
            for re in node_in_out[node][direction]:
                for la in node_in_out[node][direction][re]:
                    if not label_freq_dict[direction].get(re):
                        label_freq_dict[direction][re] = dict()
                    if not label_freq_dict[direction][re].get(la):
                        label_freq_dict[direction][re][la] = 0
                    label_freq_dict[direction][re][la] += 1/node_count

    complete_label_graph = list() # label对应的完整性子图
    comparison_dict = {'in': set(), 'out': set()} # 比较dict，用于寻找有缺失的结点
    missing_situations = dict() # label对应结点的缺失情况dict

    # 计算complete_label_graph与comparison_dict
    for direction in label_freq_dict.keys():
        if direction == 'in':
            for re in label_freq_dict[direction].keys():
                for la in label_freq_dict[direction][re].keys():
                    if label_freq_dict[direction][re][la] > complete_rate:
                        complete_label_graph.append({'head': la, 'head_typ':la, 'rel':re, 'tail':label, 'tail_typ':label})
                        comparison_dict[direction].add(re)
        if direction == 'out':
            for re in label_freq_dict[direction].keys():
                for la in label_freq_dict[direction][re].keys():
                    if label_freq_dict[direction][re][la] > complete_rate:
                        complete_label_graph.append({'head': label, 'head_typ':label, 'rel': re, 'tail': la, 'tail_typ':la})
                        comparison_dict[direction].add(re)

    # 计算missing_situations
    for node in node_in_out.keys():
        missing_situations[node] = []
        for direction in node_in_out[node]:
                if direction == 'in':
                    for re_need in comparison_dict[direction]:
                        if re_need not in node_in_out[node][direction].keys():
                            missing_situations[node].append({'head':'unknown', 'head_typ':'unknown', 'rel': re_need, 'tail': node, 'tail_typ': label})
                if direction == 'out':
                    for re_need in comparison_dict[direction]:
                        if re_need not in node_in_out[node][direction].keys():
                            missing_situations[node].append({'head': node,'head_typ': label, 'rel': re_need, 'tail':'unknown', 'tail_typ': 'unknown'})

    missing_situations_table = list() # 返回给前端的table
    for node in missing_situations.keys():
        missing_situations_table.append({
                'label': label,
                'ent' : node,
                'missing_count' : len(missing_situations[node]),
                'missing_tuple' : missing_situations[node]
            })
    missing_situations_table.sort(key=lambda d: d['missing_count'], reverse=True)

    return complete_label_graph, missing_situations_table