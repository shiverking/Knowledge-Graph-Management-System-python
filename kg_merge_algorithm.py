import re
# 描述：计算两个中文语句的相似度，这里给出四种方法
import jieba  # jieba分词
import difflib  # 方法一：Python自带标准库计算相似度的方法，可直接用
from fuzzywuzzy import fuzz  # 方法二：Python自带标准库计算相似度的方法，可直接用
import numpy as np
from collections import Counter
import Levenshtein

# 最小编辑距离(Levenshtein)
def edit_similar(str1, str2):
    return Levenshtein.ratio(str1, str2)

# 余弦相似度
def cos_sim(str1, str2):  # str1，str2是分词后的标签列表
    str1 = jieba.lcut(str1)
    str2 = jieba.lcut(str2)
    co_str1 = (Counter(str1))
    co_str2 = (Counter(str2))
    p_str1 = []
    p_str2 = []
    for temp in set(str1 + str2):
        p_str1.append(co_str1[temp])
        p_str2.append(co_str2[temp])
    p_str1 = np.array(p_str1)
    p_str2 = np.array(p_str2)
    return p_str1.dot(p_str2) / (np.sqrt(p_str1.dot(p_str1)) * np.sqrt(p_str2.dot(p_str2)))

def difflib_sim(str1,str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def fuzz_sim(str1,str2):
    return fuzz.ratio(str1, str2) / 100

# 计算最终成绩
def final_sore(str1,str2,param_diff=0.3,param_fuzz=0.3,param_edit=0.1,param_cos=0.3):
    return param_diff*difflib_sim(str1,str2)+param_fuzz*(fuzz_sim(str1, str2) / 100)+param_edit*edit_similar(str1, str2)+param_cos*cos_sim(str1, str2)

#数据处理
def process(sentence):
    r1 = "[“ ” _ . ! + - = —— , $ % ^ ，。？、~ @ # ￥ % …… & *《》<>「」{}【】()（）/]"
    r2 = "[\\\[\]'\"]"
    sentence = re.sub(r1, '', sentence)
    sentence = re.sub(r2, '', sentence)
    return sentence
