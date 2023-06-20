from pyMysql import mysql_query, mysql_in_up_re
import time

def get_time():
    t = time.localtime()
    return f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}-{t.tm_sec}' 

def get_modSet_for_selection():
    '''查找所有的链接预测模型'''
    sql = '''select id, model_name 
            from link_prediction_model 
            where train_status=0'''
    res = list()
    sql_res = mysql_query(sql)
    for model in sql_res:
        res.append({'value': model[0], 'label': model[1]})
    return res

def add_model_to_lpm(model_name):
    '''新增的模型存入数据库'''
    model_address = 'ConvE\models'
    sql = 'insert into link_prediction_model(model_name, model_address, train_status, create_time) values(%s, %s, %s, %s)'
    res = mysql_in_up_re(sql, ({model_name}, {model_address}, 1, get_time()))
    return res

def get_lpms():
    '''获取模型列表'''
    sql = '''select model_name, model_address, train_status, create_time 
                from link_prediction_model 
                order by create_time desc'''
    res = list()
    sql_res = mysql_query(sql)
    for model in sql_res:
        res.append({'model_name': model[0], 'model_address': model[1], 'train_status':model[2], 'create_time': str(model[3])})
    return res

def lpm_finish(name):
    '''训练完成后将模型的训练状态设为0'''
    sql = 'select id from link_prediction_model where model_name = (%s)'
    sql_res = mysql_query(sql, name)
    sql = "update link_prediction_model set train_status = 0 where id = (%s)"
    mysql_in_up_re(sql, sql_res[0][0])

def delete_lpm(name):
    '''删除模型'''
    sql = 'select id from link_prediction_model where model_name = (%s)'
    sql_res = mysql_query(sql, name)
    sql = "delete from link_prediction_model where id = (%s)"
    res = mysql_in_up_re(sql, sql_res[0][0])
    return res

if __name__ == '__main__':
    print(lpm_finish('sahgdashj'))