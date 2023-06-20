from flask import Flask,jsonify,request
from flask_cors import CORS
from ConvE.conve import link_prediction_1
from helper import get_gpu_info, get_abnormal_by_kgist, get_entSet_for_selection, get_relSet_for_selection, a_jump_of_hr, \
        link_completion, links_notConform_ontology, search_type_of_ent, sample_from_CSG
from linkPrediction2Mysql import get_modSet_for_selection, get_lpms, add_model_to_lpm, get_lpms, lpm_finish, delete_lpm
from ConvE.conve import triple_classification as triple_confidence
from ConvE.conve import train_model
import psutil
from entityAlignmentService import calSimilarity,calSimilarityFromCoreKg
from entityController import getAlLEntites
from mysql2neo4j import insert2neo4j, select_synchronization_from_version_record, update_synchronization_from_version_record
from integrity_checks import get_the_complete_label_graph_and_missing_situations, tuple_integrity_graph
from casrel.main import extract_triples
from ch_triple_extraction import ch_tri_ext

server = Flask(__name__)
server.config['JSON_AS_ASCII']=False

entites, relations = [], []

@server.route('/link_prediction_demo',methods=['post'])
def link_prediction_demo():
    '''待预测数据随机采样-演示'''
    res = sample_from_CSG()
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/triple_extraction_demo',methods=['post'])
def triple_extraction_demo():
    '''三元组抽取-演示'''
    inputs = request.json.get('data')
    res = ch_tri_ext(inputs)
    dict = {}
    dict['data'] = res
    dict['status'] = 200
    return jsonify(dict)

@server.route('/triple_extraction',methods=['post'])
def triple_extraction():
    '''三元组抽取服务'''
    inputs = request.json.get('data')
    res = extract_triples(inputs)
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/link_prediction',methods=['post'])
def link_prediction():
    '''链接预测服务'''
    inputs = request.json.get('data')
    print(inputs)
    need_to_predict = []
    model_name = inputs['model_name']
    for input in inputs['ent_and_rel']:
        need_to_predict.append((input['ent'], input['rel']))
    res = link_prediction_1(need_to_predict, model_name)
    for idx in range(len(res['preds'])):
        for in_idx in range(len(res['preds'][idx])):
            new_list = list(res['preds'][idx][in_idx])
            new_list.append(search_type_of_ent(res['preds'][idx][in_idx][0]))
            res['preds'][idx][in_idx] = tuple(new_list)
    dict = {}
    dict['data'] = res
    print(dict)
    return jsonify(dict)

@server.route('/triple_classification',methods=['post'])
def triple_classification():
    '''三元组置信度检测服务'''
    inputs = request.json.get('data')
    res = triple_confidence(inputs)
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/train_new_model',methods=['post'])
def train_new_model():
    '''训练新的链接预测模型'''
    inputs = request.json.get('data')
    model_name = inputs['name']
    res = train_model(model_name)
    print("train_new_model的结果：",res)
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/add_model_to_lpm_table',methods=['post'])
def add_model_to_lpm_table():
    '''增加链接预测模型记录到数据库'''
    inputs = request.json.get('data')
    model_name = inputs['name']
    add_model_to_lpm(model_name)
    dict = {}
    dict['data'] = ''
    return dict

@server.route('/learning_finish',methods=['post'])
def learning_finish():
    '''链接预测训练结束-修改train_status为0'''
    model_name = request.json.get('data')
    res = lpm_finish(model_name)
    dict = {}
    dict['data'] = res
    return dict

@server.route('/get_status_of_cpu',methods=['post'])
def get_status_of_cpu():
    '''获取cpu状态'''
    res = [psutil.cpu_percent()]
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/get_status_of_gpu',methods=['post'])
def get_status_of_gpu():
    '''获取gpu状态'''
    res = [get_gpu_info()]
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/get_relation_error_list',methods=['post'])
def get_relation_error_list():
    '''获取链接异常列表'''
    res = links_notConform_ontology()
    dict = {}
    dict['data'] = res
    return jsonify(dict)

# @server.route('/get_attribute_error_list',methods=['post'])
# def get_attribute_error_list():
#     res = attribute_error_simulation()
#     dict = {}
#     dict['data'] = res
#     return jsonify(dict)

@server.route('/calculateEntitySimilarity',methods=['post'])
def calculate_the_entity_similarity():
    '''计算实体相似度'''
    firstEntity = request.json.get('firstEntity')
    secondEntity = request.json.get('secondEntity')
    threshold = float(request.json.get('threshold'))
    algorithm = request.json.get('algorithm')
    result = calSimilarity(algorithm,threshold,firstEntity,secondEntity)
    if len(result)>0:
        result = {"code":"200","count":len(result),"msg":"success","data":result}
        return jsonify(result)
    else:
        result = {"code": "200", "count":0,"msg": "finish", "data": []}
        return jsonify(result)

@server.route('/calculateEntitySimilarityFromCoreKg',methods=['post'])
def calculate_the_entity_similarity_corekg():
    '''针对图谱的实体对齐'''
    entites, relations = getAlLEntites('42.192.6.2',34235,'root','Sicdp2021fkfd@','kgms_test')
    secondEntity = request.json.get('secondEntity')
    threshold = float(request.json.get('threshold'))
    algorithm = request.json.get('algorithm')
    result = calSimilarityFromCoreKg(algorithm,threshold,entites,secondEntity)
    if len(result)>0:
        result = {"code":"200","count":len(result),"msg":"success","data":result}
        return jsonify(result)
    else:
        result = {"code": "200", "count":0,"msg": "finish", "data": []}
        return jsonify(result)    

@server.route('/get_abnormal_by_algorithm',methods=['post'])
def get_abnormal_by_algorithm():
    '''算法检测到的链接异常'''
    res = get_abnormal_by_kgist()
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/get_entSet',methods=['post'])
def get_entSet():
    '''读取实体集合-用于链接预测'''
    res = get_entSet_for_selection()
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/get_relSet',methods=['post'])
def get_relSet():
    '''读取关系集合-用于链接预测'''
    res = get_relSet_for_selection()
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/get_modSet',methods=['post'])
def get_modSet():
    '''读取模型集合-用于链接预测'''
    res = get_modSet_for_selection()
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/get_lpm_table',methods=['post'])
def get_lpm_table():
    '''读取模型集合-用于展示'''
    res = get_lpms()
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/draw_graph',methods=['post'])
def draw_graph():
    '''展示三元组周围一跳的信息'''
    inputs = request.json.get('data')
    res = a_jump_of_hr(inputs)
    dict = {}
    dict['data'] = res
    return jsonify(dict)

# @server.route('/draw_graph_of_attr',methods=['post'])
# def draw_graph_of_attr():
#     inputs = request.json.get('data')
#     res = a_jump_of_ea(inputs)
#     dict = {}
#     dict['data'] = res
#     return jsonify(dict)

@server.route('/remove_lpm',methods=['post'])
def remove_lpm():
    '''删除链接预测模型'''
    inputs = request.json.get('data')
    res = delete_lpm(inputs)
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/coreKG2neo4j',methods=['post'])
def coreKG2neo4j():
    '''coreKG的数据插入neo4j'''
    if select_synchronization_from_version_record():
        update_synchronization_from_version_record()
        insert2neo4j()
    dict = {}
    dict['data'] = 'true'
    return jsonify(dict)

@server.route('/linked_completion',methods=['post'])
def linked_completion():
    '''链接补全结果'''
    res = link_completion()
    dict = {}
    dict['data'] = res
    return jsonify(dict)

@server.route('/return_integrity_results',methods=['post'])
def return_integrity_results():
    '''链接完整性检测结果'''
    label_name = request.json.get('data')
    rate = request.json.get('rate')
    print(label_name, rate)
    complete_label_graph, missing_situations_table = get_the_complete_label_graph_and_missing_situations(label_name, rate)
    dict = {}
    dict['complete_label_graph'] = complete_label_graph
    dict['missing_situations_table'] = missing_situations_table
    return jsonify(dict)

@server.route('/return_tuple_integrity_graph',methods=['post'])
def return_tuple_integrity_graph():
    '''链接完整性检测结果'''
    ent = request.json.get('ent')
    ent_typ = request.json.get('ent_typ')
    missing_tuple = request.json.get('missing_tuple')
    dict = {}
    dict['tuple_integrity_graph'] = tuple_integrity_graph(ent, ent_typ, missing_tuple)
    return jsonify(dict)

if __name__ == "__main__":
    ip="0.0.0.0"
    port = 3389
    CORS(server, supports_credentials=True)
    server.run(host = ip, port = port, use_reloader=False)