from py2neo import Graph, Node, Relationship, Graph, NodeMatcher, RelationshipMatcher
import pandas as pd

# 读excel
df = pd.read_excel('/home/cjw/KGtest/triples.xls')
row_num = len(df.index.values)
print("三元组数：", row_num)
# 去掉有缺失数据
df.dropna(axis=0, how='any', inplace=True)
row_num = len(df.index.values)
print("三元组数：", row_num)
triples = df.values.tolist()
triples = [triple[:-1] for triple in triples]

graph = Graph('http://192.168.100.114:7474', auth=('neo4j', 'Sicdp2021cjw'))

# 查找实体类型对应实例，返回list
def get_all_entities_of_ent_typ(graph, ent_typ):
    matcher = NodeMatcher(graph)
    ent_list = list(matcher.match(ent_typ))
    ent_list = [ent['name'] for ent in ent_list]
    return ent_list

# 查找所有的实体类型
def get_all_entity_types(graph):
    return list(graph.schema.node_labels)

# 查找所有的关系类型
def get_all_relationship_types(graph):
    return list(graph.schema.relationship_types)

# 三元组插入neo4j
def triples2neo4j(graph, triples, one2many=False, many2one=False):
    for triple in triples:
        # 取出头实体、尾实体、关系
        # ent_1, ent_2, rel = triple
        # head, head_typ = ent_1
        head, head_typ, tail, tail_typ, rel = triple
        head_node = Node(head_typ, name=head)
        # tail, tail_typ = ent_2
        tail_node = Node(tail_typ, name=tail)
        # head类型list
        head_list = get_all_entities_of_ent_typ(graph, head_typ)
        # tail类型list
        tail_list = get_all_entities_of_ent_typ(graph, tail_typ)
        # 头实体和尾实体都存在 """暂且不判断正误"""
        if head in head_list and tail in tail_list:
            graph.merge(head_node, head_typ, "name")
            graph.merge(tail_node, tail_typ, "name")
            if list(RelationshipMatcher(graph).match((head_node, tail_node), r_type = rel)):
                print(f'三元组 ({head} ,{tail} ,{rel}) 已存在于图谱中，插入失败！')
            else:
                graph.create(Relationship(head_node, rel, tail_node))
                print(f'三元组 ({head} ,{tail} ,{rel}) 插入成功！')
        # 头实体已存在 """暂且不判断正误"""
        elif head in head_list and tail not in tail_list:
            graph.merge(head_node, head_typ, "name")
            if list(RelationshipMatcher(graph).match((head_node, None), r_type = rel)):
                if one2many == False:
                    print(f'头实体 {head} 已存在关系 {rel} 对应的三元组 ({head} ,{tail} ,{rel})，插入失败！')
                    continue
            graph.create(tail_node)
            graph.create(Relationship(head_node, rel, tail_node))
            print(f'三元组 ({head} ,{tail} ,{rel}) 插入成功！')
        # 尾实体已存在 """暂且不判断正误"""
        elif head not in head_list and tail in tail_list:
            graph.merge(tail_node, tail_typ, "name")
            if list(RelationshipMatcher(graph).match((None, tail_node), r_type = rel)):
                if many2one == False:
                    print(f'尾实体 {tail} 已存在关系 {rel} 对应的三元组 ({head} ,{tail} ,{rel})，插入失败！')   
                    continue             
            graph.create(head_node)
            graph.create(Relationship(head_node, rel, tail_node))
            print(f'三元组 ({head} ,{tail} ,{rel}) 插入成功！')
        # 头实体、尾实体均不存在
        else:                    
            graph.create(head_node)
            graph.create(tail_node)
            graph.create(Relationship(head_node, rel, tail_node))
            print(f'三元组 ({head} ,{tail} ,{rel}) 插入成功！')

# triples = [
#     (['李沐','Per'], ['CMU', 'Sch'], '毕业于'),
#     (['李沐', 'Per'], ['沐神的小迷弟', 'Per'], '迷弟'),
#     (['李沐','Per'], ['中国', 'Cou'], '出生于'),
#     (['李沐','Per'], ['亚马逊', 'Com'], '就职于'),
#     (['沐神的小迷弟', 'Per'], ['西安交通大学', 'Sch'], '就读于'),
#     (['李沐','Per'], ['上海交通大学', 'Sch'], '毕业于'),
#     (['李沐','Per'], ['百度', 'Com'], '就职于'),
#         ]

triples2neo4j(graph, triples, one2many=True, many2one=True)

