import pickle

import werkzeug
from flask_restful import reqparse, abort, Resource
from flask import current_app

from app.wrapper import ShapeMismatchError, NotImageError


class ClassifyImage(Resource):
    """
    API Resource, providing image classification, using the fitted model
    """

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads model, according to the application configuration
        :return: Model
        """
        with open(current_app.config['MODEL_LOCATION'], 'rb') as handle:
            return pickle.load(handle)

    def post(self):
        """
        POST-end point. Accepts image as "image" parameter in form data.
        :return: Dictionary with response
        """
        parser = reqparse.RequestParser()
        parser.add_argument('image', type=werkzeug.datastructures.FileStorage,
                            location='files',
                            required=True,
                            help="Image is required")

        args = parser.parse_args()
        stream = args['image'].stream
        try:
            predicted_class = self.model.predict(stream)
            return {'class': predicted_class}
        except NotImageError:
            abort(400, description="The given file is not an image")
        except ShapeMismatchError as e:
            abort(400, description="Image shape contradicts the model. Expected: {}, Actual: {}"
                  .format(e.expected, e.actual))