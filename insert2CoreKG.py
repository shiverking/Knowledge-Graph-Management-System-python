from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import datetime
from urllib.parse import quote_plus as urlquote

app = Flask(__name__)
# MySQL所在主机名
HOSTNAME = "42.192.6.2"
# MySQL监听的端口号，默认3306
PORT = 34235
# 连接MySQL的用户名，自己设置
USERNAME = "root"
# 连接MySQL的密码，自己设置
PASSWORD = "Sicdp2021fkfd@"
# MySQL上创建的数据库名称
DATABASE = "kgms_test"
# 通过修改以下代码来操作不同的SQL比写原生SQL简单很多 --》通过ORM可以实现从底层更改使用的SQL
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{USERNAME}:{urlquote(PASSWORD)}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class CoreKG(db.Model):
    __tablename__ = 'core_kg'
    head = db.Column(db.String(64),primary_key=True)
    relation = db.Column(db.String(64),primary_key=True)
    tail = db.Column(db.String(64),primary_key=True)
    time = db.Column(db.DateTime, default=datetime.datetime.now)
    tail_type = db.Column(db.String(64))
    head_type = db.Column(db.String(64))
    version = db.Column(db.String(64))

def add_tuple_to_ck(tuples):
    with app.app_context():
        for tuple in tuples:
            db.session.add(CoreKG(
            head = tuple['head'],
            relation = tuple['relation'],
            tail = tuple['tail'],
            tail_type = tuple['tail_type'],
            head_type = tuple['head_type'],
            version = tuple['version'],
            ))
        db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        model_list = db.session.query(CoreKG.head, CoreKG.head_type).all()
        print(model_list)