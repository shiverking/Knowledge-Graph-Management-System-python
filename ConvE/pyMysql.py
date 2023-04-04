import pymysql

def connect_mysql():
    """连接mysql数据库"""
    conn = pymysql.connect(
        host='42.192.6.2',
        port=34235,
        user='root',
        password='Sicdp2021fkfd@',
        db='kgms_test',
        charset='utf8')
    #创建游标
    cursor=conn.cursor() #执行完毕返回的结果默认以元组的形式保存
    return conn,cursor
 
def colse_mysql_conn(conn,cursor):
    """关闭数据库的连接"""
    cursor.close()
    conn.close()
  
def mysql_query(sql,*args):
    """封装通用查询"""
    conn,cursor=connect_mysql() #连接数据库，创建游标
    cursor.execute(sql,args) #执行sql语句
    res=cursor.fetchall() #获取返回的数据（元组形式）
    colse_mysql_conn(conn,cursor) #关闭游标，以及数据库连接
    return res

def mysql_in_up_re(sql, *args):
    """封装增删改"""
    isSucess = False
    conn, cursor=connect_mysql()
    try:
        cursor.executemany(sql, args)
        conn.commit()
        isSucess = True
    except:
        conn.rollback()
    colse_mysql_conn(conn,cursor)
    return isSucess