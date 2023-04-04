#发布http服务，并且注册到nacos
from flask import Flask,jsonify,request
from service.entityAlignmentService import calSimilarity
import requests
import time

# Flask初始化参数尽量使用你的包名，这个初始化方式是官方推荐的，官方解释：http://flask.pocoo.org/docs/0.12/api/#flask.Flask
server = Flask(__name__)
#处理乱码
server.config['JSON_AS_ASCII']=False

@server.route('/calculateEntitySimilarity',methods=['post'])
def calculate_the_entity_similarity():
    firstEntity = request.json.get('firstEntity')
    secondEntity = request.json.get('secondEntity')
    threshold = float(request.json.get('threshold'))
    algorithm = request.json.get('algorithm')
    result = calSimilarity(algorithm,threshold,firstEntity,secondEntity)
    if len(result)>0:
        result = {"code":"200","count":len(result),"msg":"success","data":result}
        return jsonify(result)
    else:
        result = {"code": "200", "count":0,"msg": "finish", "data": []}
        return jsonify(result)

if __name__ == "__main__":
    ip = "127.0.0.1"
    port=8088
    server.run(port=port,debug=True)