import pickle

import imageio
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import DatasetSample
from sklearn.linear_model.base import LinearModel


class ShapeMismatchError(Exception):
    """
    Is raised, when array of different shape was expected.
    """

    def __init__(self, actual, expected, *args, **kwargs):
        super(ShapeMismatchError).__init__(*args, **kwargs)
        self._actual = actual
        self._expected = expected

    @property
    def actual(self):
        return self._actual

    @property
    def expected(self):
        return self._expected


class NotImageError(Exception):
    """
    Is raised, when given file is not an image.
    """
    pass


class ModelServingWrapper:
    """
    Wrapper for sklearn.linear_model.base.LinearModel. Simplifies training on samples from the database and
    prediction, accepting images and handling typical exceptions.
    """

    def __init__(self, model: LinearModel, db_path: str):
        """
        Inits the wrapper.
        :param model: Unfitted model
        :param db_path: SQLAlchemy-compatible path to the database
        """
        if hasattr(model, 'coef_') and model.coef_ is not None:
            raise ValueError("Could not create wrapper for already trained model")

        self._db_path = db_path
        self._model = model
        self._item_shape = None

    def fit(self):
        """
        Fits the model, using samples from the database and returns accuracy
        :return: Accuracy of the fitted model on the training set
        """
        image, labels = self._fit()
        return self._estimate_accuracy(image, labels)

    def _fit(self):
        """
        Fits the model, using samples from the database
        :return: Images and labels ndarray, built from the database
        """
        images, labels, item_shape = self._get_training_dataset()
        self._model.fit(images, labels)
        self._item_shape = item_shape
        return images, labels

    def _estimate_accuracy(self, images, labels):
        """
        Estimates accuracy of the model on images and labels
        :param images: Samples
        :param labels: True labels for samples
        :return: Accuracy
        """
        predicted_labels = self._model.predict(images)
        return accuracy_score(labels, predicted_labels)

    @staticmethod
    def _image_to_ndarray(image):
        """
        Tries to convert given image to ndarray
        :param image: Image, in any imageio-compatible format
        :return: ndarray
        """
        try:
            return imageio.imread(image)
        except ValueError:
            raise NotImageError()

    def predict(self, image_or_vector):
        """
        Predicts the class for given data
        :param image_or_vector: Image, which could be read by imageio library or ndarray
        :return: int, class label
        """

        if type(image_or_vector) is not np.ndarray:
            image_or_vector = self._image_to_ndarray(image_or_vector)

        if self._item_shape is None:
            raise NotFittedError("Model is not fitted yet")

        if image_or_vector.shape != self._item_shape:
            raise ShapeMismatchError(image_or_vector.shape, self._item_shape)

        return int(self._model.predict(image_or_vector.reshape(1, -1))[0])

    def _get_training_dataset(self):
        """
        Retrieves dataset items from the database as ndarray with flatten images,
        ndarray with labels and image shape.
        :return: ndarray with images, ndarray with labels, one image shape
        """
        engine = create_engine(self._db_path)
        DBSession = sessionmaker(bind=engine)
        session = DBSession()
        items = session.query(DatasetSample).all()

        item_shape = None

        images = []
        labels = []

        for item in items:
            image = pickle.loads(item.image)
            if item_shape is None:
                item_shape = image.shape
            elif item_shape != image.shape:
                raise ValueError("Shape mismatch in training database")
            images.append(image.flatten())
            labels.append(item.label)

        if len(items) > 0:
            return np.vstack(images), np.array(labels), item_shape
        else:
            raise ValueError("The database is empty")
