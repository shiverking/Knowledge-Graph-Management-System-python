    
from KGist.src.anomaly_detector import AnomalyDetector
import pickle
def get_abnormal_by_kgist():
    file = open('E:\\KG_system_service\\KGist\\output\\aircra_model_Rm_Rn.pickle','rb')  # 以二进制读模式（rb）打开pkl文件
    model = pickle.load(file)
    detector = AnomalyDetector(model)
    delimiter = ','
    score_list = dict()
    edgelist = 'E:\\KG_system_service\\KGist\\data\\aircra.txt'
    labellist= 'E:\\KG_system_service\\KGist\\data\\aircra_labels.txt'
    with open(edgelist, 'r', encoding='utf-8') as f:
        for line in f:
            sub, pred, obj = line.strip().split(delimiter)
            # print(f'the score_edge of {sub} {pred} {obj} is {detector.score_edge((sub, pred, obj))}')
            score_list[(sub, pred, obj)] = detector.score_edge((sub, pred, obj))
    plot_list = sorted(score_list.items(), key=lambda d: d[1], reverse=True)
    with open(labellist, 'r', encoding='utf-8') as f:
        ent_to_label = dict()
        for line in f:
            ent, label = line.strip().split(delimiter)[:2]
            ent_to_label[ent] = label
    res = list()
    for key, value in plot_list:
        head, rel, tail = key
        res.append({'head':head, 'head_typ':ent_to_label[head], 'rel':rel, 'tail':tail, 'tail_typ':ent_to_label[tail], 'abnormal_score':[value]})
    return res

def save_as_pickle(a_list_or_dict, save_name):
    save_file = open(save_name + '.pkl', 'wb')
    pickle.dump(a_list_or_dict, save_file)
    save_file.close()

def load_pickle_file(file_path):
    a_list_or_dict = open(file_path, 'rb')
    return pickle.load(a_list_or_dict)

if __name__ == '__main__':
    res = get_abnormal_by_kgist()
    save_as_pickle(res, 'aircra_abnormal')
    print(load_pickle_file('aircra_abnormal.pkl'))