from kg_merge_algorithm import *
from Jaccard import Match_list
from kg_merge_algorithm import *
#前端传过来的两个实体列表
def calSimilarity(algorithm,threshold,toList,fromList):
    toEntity = []
    fromEntity = []
    result = []
    for item in toList:
        toEntity.append([item['head'],process(item['head']),item['id']])
        toEntity.append([item['tail'],process(item['tail']),item['id']])
    for item in fromList:
        fromEntity.append([item['head'],process(item['head']),item['id']])
        fromEntity.append([item['tail'],process(item['tail']),item['id']])
    stored_pair=[]
    if algorithm=="最小编辑距离":
        for item1 in toEntity:
            for item2 in fromEntity:
                sim = edit_similar(item1[1],item2[1])
                if sim >= threshold and sim<1:
                    new_record  = {"to":item1[0],"from":item2[0],"sim":sim}
                    if new_record not in stored_pair:
                        stored_pair.append(new_record)
                        result.append({"toId":item1[2],"to":item1[0],"from":item2[0],"fromId":item2[2],"sim":sim})
    elif algorithm == "余弦相似度":
        for item1 in toEntity:
            for item2 in fromEntity:
                sim = cos_sim(item1[1],item2[1])
                if sim >= threshold and sim<1:
                    new_record = {"to": item1[0], "from": item2[0], "sim": sim}
                    if new_record not in stored_pair:
                        stored_pair.append(new_record)
                        result.append({"toId": item1[2], "to": item1[0], "from": item2[0], "fromId": item2[2], "sim": sim})
    elif algorithm == "Difflib":
        for item1 in toEntity:
            for item2 in fromEntity:
                sim = difflib_sim(item1[1],item2[1])
                if sim >= threshold and sim<1:
                    new_record = {"to": item1[0], "from": item2[0], "sim": sim}
                    if new_record not in stored_pair:
                        stored_pair.append(new_record)
                        result.append({"toId": item1[2], "to": item1[0], "from": item2[0], "fromId": item2[2], "sim": sim})
    elif algorithm == "Fuzzywuzzy":
        for item1 in toEntity:
            for item2 in fromEntity:
                sim = fuzz_sim(item1[1],item2[1])
                if sim >= threshold and sim<1:
                    new_record = {"to": item1[0], "from": item2[0], "sim": sim}
                    if new_record not in stored_pair:
                        stored_pair.append(new_record)
                        result.append({"toId": item1[2], "to": item1[0], "from": item2[0], "fromId": item2[2], "sim": sim})
    elif algorithm == "加权混合":
        for item1 in toEntity:
            for item2 in fromEntity:
                sim = final_sore(item1[1],item2[1])
                if sim >= threshold and sim<1:
                    new_record = {"to": item1[0], "from": item2[0], "sim": sim}
                    if new_record not in stored_pair:
                        stored_pair.append(new_record)
                        result.append({"toId": item1[2], "to": item1[0], "from": item2[0], "fromId": item2[2], "sim": sim})
    return result
#将传来的图谱和核心图谱中的的数据进行计算
def calSimilarityFromCoreKg(algorithm,threshold,entities,fromList):
    fromEntity = []
    result = []
    for item in fromList:
        fromEntity.append([item['head'],process(item['head']),item['id']])
        fromEntity.append([item['tail'],process(item['tail']),item['id']])
    stored_pair=[]
    if algorithm=="Jaccard相似系数":
        for item1 in fromEntity:
            # 匹配
            match_list = Match_list(item1[1], entities);
            res = match_list.approximate_match(threshold*100);
            for item2 in res:
                new_record  = {"to":item2[0],"from":item1[0],"sim":item2[1]}
                if new_record not in stored_pair:
                    stored_pair.append(new_record)
                    result.append({"to":item2[0],"from":item1[0],"fromId":item1[2],"sim":item2[1]})
    result.sort(key=lambda x: x['sim'], reverse=True)
    return result