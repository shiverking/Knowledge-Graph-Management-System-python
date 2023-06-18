    
from anomaly_detector import AnomalyDetector
import pickle
from pyMysql import mysql_query
def kgist_predict():
    file = open('output/demo_model.pickle','rb')  # 以二进制读模式（rb）打开pkl文件
    model = pickle.load(file)
    detector = AnomalyDetector(model)
    score_list = dict()
    edgelist = 'data/demo.txt'
    sql = 'select head, head_type, relation, tail, tail_type from core_kg'
    query_res = mysql_query(sql)
    for sub, _, pred, obj, tail_typ in query_res:
        if tail_typ != 'value':
            score_list[(sub, pred, obj)] = detector.score_edge((sub, pred, obj))
    plot_list = sorted(score_list.items(), key=lambda d: d[1], reverse=True)
    sql = 'select head, head_type from core_kg union select tail, tail_type from core_kg'
    query_res = mysql_query(sql)
    ent_to_label = dict()
    for line in query_res:
        if line[1] != 'value':
            ent_to_label[line[0]] = line[1]
    res = list()
    for key, value in plot_list:
        head, rel, tail = key
        res.append({'head':head, 'head_typ':ent_to_label[head], 'rel':rel, 'tail':tail, 'tail_typ':ent_to_label[tail], 'abnormal_score':[value],\
                    'error_status': 0, 'error_typ': 0})
    return res