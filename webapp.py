from flask import Flask
from flask_restful import Api

from app.resources import ClassifyImage

app = Flask(__name__)
api = Api(app)

app.config.from_object('config.Default')
api.add_resource(ClassifyImage, '/api/classify')


if __name__ == '__main__':
    app.run()
