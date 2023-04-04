#发布http服务，并且注册到nacos
from flask import Flask,jsonify,request
import requests
import threading
import time
from nacos import service_register, service_beat, get_public_ip

# Flask初始化参数尽量使用你的包名，这个初始化方式是官方推荐的，官方解释：http://flask.pocoo.org/docs/0.12/api/#flask.Flask
server = Flask(__name__)
#处理乱码
server.config['JSON_AS_ASCII']=False
'''使用restful进行get请求，通过请求地址进行传参，其中胡涛是传参
请求地址：http://127.0.0.1:8085/simulation/analysis/胡涛
请求参数：将地址中的胡涛映射到属性字段name上
响应参数：
{
    "code": "200",
    "data": {
        "age": 25,
        "job": "python",
        "name": "胡涛"
    },
    "msg": "SUCCES"
}
'''
# @server.route('/simulation/analysis/<name>',methods=['get'])
# def demo_restful_request(name):
#     # 处理业务逻辑
#     name = request.args['name']
#     result = {"code":"200","msg":"SUCCES","data":{"name":name,"age":25,"job":"python"}}
#     return jsonify(result)

'''使用rest进行get请求，通过请求拼接参数进行传参，其中name是传参
请求实例：http://127.0.0.1:8085/simulation/analysis?name=胡涛
请求参数：请求地址中的name=胡涛
响应参数
{
    "code": "200",
    "data": {
        "age": 25,
        "job": "python",
        "name": "胡涛"
    },
    "msg": "SUCCES"
}
'''
@server.route('/123',methods=['get','post'])
def demo_rest_get_request():
    print("有人调用")
    # 处理业务逻辑
    name = request.args['name']
    result = {"code":"200","msg":"SUCCES","data":{"name":name,"age":25,"job":"python"}}
    return jsonify(result)


'''使用rest进行post请求，通过请求提 json传参，其中name是传参
请求地址：http://127.0.0.1:8085/simulation/analysis
请求参数：
{
    "name":"胡涛",
    "job":"java"
}
响应参数：
{
    "code": "200",
    "data": {
        "age": 25,
        "job": "java",
        "name": "胡涛"
    },
    "msg": "SUCCES"
}

'''
@server.route('/post/body',methods=['post'])
def demo_rest_post_request():
    # 处理业务逻辑
    name = request.json.get('name')
    job = request.json.get('job')
    result = {"code":"200","msg":"SUCCES","data":{"name":name,"age":25,"job":job}}
    return jsonify(result)
if __name__ == "__main__":
    ip = "127.0.0.1"
    port=8088
    # register_url = service_register(ip=ip,port=port,serviceName='algorithm-service',namespaceId="3928050f-9850-487d-91bb-8859f48d48b5",groupName='1')
    # register_url = service_register(ip=ip, port=port, serviceName='algorithm-service')
    # print(register_url)
    #5秒以后，异步执行service_beat()方法
    # threading.Timer(5,service_beat(register_url)).start()
    server.run(port=port,debug=True)