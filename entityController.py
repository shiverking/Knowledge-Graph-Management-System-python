import pymysql

# 获取所有实体
def getAlLEntites(host,port,user,password,database):
    db = pymysql.connect(host=host,user=user,password=password,port=port,database=database,charset='utf8')
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("SELECT head FROM core_kg union SELECT tail FROM core_kg")

    # 使用 fetchAll() 方法获取所有实体
    data = cursor.fetchall()
    entites=[]
    for item in data:
        entites.append(item[0])
    print("实体加载成功")

    # 使用execute()方法执行SQL查询所有关系
    cursor.execute("SELECT relation FROM core_kg")
    rel = cursor.fetchall()
    relations = []
    for item in rel:
        relations.append(item[0])
    print("关系加载成功")

    # 关闭数据库连接
    db.close()
    return entites,relations