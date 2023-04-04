from pyMysql import mysql_query
from py2neo import Graph, Node, Relationship, Graph

def select_synchronization_from_version_record():
    '''查询版本管理中的未更新的版本号'''
    sql = 'select count(*) from version_record where synchronization = 0'
    res = mysql_query(sql)
    if res[0][0] != 0:
        return True
    else:
        return False

def update_synchronization_from_version_record():
    '''更新所有的版本状态为1'''
    sql = 'update version_record set synchronization = 1'
    import pymysql
    try:
        db = pymysql.connect(host='42.192.6.2',port=34235,user='root',password='Sicdp2021fkfd@',database='kgms_test')
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        db.close()
    except:
        return False
    return True   

def traverse_coreKG():
    '''遍历核心图谱'''
    sql = 'select head, head_type, relation, tail, tail_type  from core_kg'
    res = mysql_query(sql)
    return res

def insert2neo4j():
    '''将核心图谱中的数据插入到图数据库'''
    graph = Graph('http://localhost:7474', auth=('neo4j', 'Sicdp2021cjw'))
    graph.delete_all() # 清空图数据库
    triples = traverse_coreKG()
    for triple in triples:
        head, head_typ, rel, tail, tail_typ = triple # 取出头实体、尾实体、关系
        head_node = Node(head_typ, name=head)
        tail_node = Node(tail_typ, name=tail)
        graph.merge(head_node, head_typ, "name")
        graph.merge(tail_node, tail_typ, "name")                  
        graph.create(Relationship(head_node, rel, tail_node))