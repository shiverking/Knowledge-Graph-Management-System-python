from algorithm.kg_merge_algorithm import *
#前端传过来的两个实体列表
def calSimilarity(algorithm,threshold,toList,fromList):
    toEntity = []
    fromEntity = []
    result = []
    for item in toList:
        toEntity.append([item['head'],process(item['head']),item['id']])
        toEntity.append([item['head'],process(item['tail']),item['id']])
    for item in fromList:
        fromEntity.append([item['head'],process(item['head']),item['id']])
        fromEntity.append([item['head'],process(item['tail']),item['id']])
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
