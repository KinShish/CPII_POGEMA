from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import main as ravel

app = Flask(__name__)
CORS(app)
api = Api(app)

class Quote(Resource):
    def get(self, size,dan,num):
        tick,svg=ravel.start(size, dan/10, num)
        return {"tick":tick,"svg":svg}, 200
class SVG(Resource):
    def get(self, name):
        return open(name, "rb").read().decode("utf-8"),200

api.add_resource(Quote, "/wtf/<int:size>/<int:dan>/<int:num>")
api.add_resource(SVG, "/svg/<string:name>")

if __name__ == '__main__':
    app.run(host='192.168.4.124',debug=True)