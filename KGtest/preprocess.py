# pandas数据处理包
import pandas as pd

# 读excel
df = pd.read_excel('/home/cjw/KGtest/triples.xls')
# print(df.groupby("relation").count())
# 获取数据行数
triples = df.values.tolist()
triples = [triple[:-1] for triple in triples]
print(triples)
