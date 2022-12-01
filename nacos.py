from flask import Flask,jsonify,request
import requests
import threading
import time
import re
#注册中心的地址，此处已经填写
ip_address ='42.192.6.2:8848'

#nacos注册中心信息

def get_public_ip():
    res = requests.get('http://myip.ipip.net', timeout=5).text.strip()
    ip = re.findall(r'\d+.\d+.\d+.\d+', res)
    return ip[0]

#nacos服务
'''
ip:本地ip地址
port:本地端口
serviceName:服务名
namespaceId:命名空间，默认'public' 不同命名空间服务互相不可见
groupId:组名默认'DEFAULT_GROUP' 不同组服务互相不可见
clusterName:集群名,默认'DEFAULT'
将服务注册到注册中心
注册说明：将http://ip:port/例如http://127.0.0.1:8085/**这个服务上的所有服务注册到注册中心，并且起名叫做serviceName 例如algorithm-service
其他微服务进行访问时，访问http://algorithm-service/**即可，即其他服务，使用algorithm-service去注册中心，寻找真实的ip地址
例如原本访问 post访问:http://127.0.0.1:8085/simulation/analysis 此时变成 http://algorithm-service/simulation/analysis
'''
def service_register(ip,port,serviceName,namespaceId='public',groupName='DEFAULT_GROUP',clusterName='DEFAULT'):
    url = "http://"+ip_address+"/nacos/v1/ns/instance?serviceName="+serviceName+"&ip="+ip+"&port="+str(port)
    if(namespaceId!='public'):
        url+='&namespaceId='+str(namespaceId)
    if (groupName != 'DEFAULT_GROUP'):
            url += '&groupName=' + str(namespaceId)
    if (clusterName != 'DEFAULT'):
        url += '&clusterName=' + str(clusterName)
    res = requests.post(url)
    print("发起nacos服务注册请求,响应状态： {}".format(res.status_code))
    return url

#服务检测
def service_beat(url):
    while True:
        res = requests.put(url)
        # print("已注册服务，执行心跳服务，续期服务响应状态： {}".format(res.status_code))
        time.sleep(5)

