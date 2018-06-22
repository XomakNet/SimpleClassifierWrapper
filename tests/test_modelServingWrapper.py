import pickle
from unittest import TestCase

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, DatasetSample
from app.wrapper import ModelServingWrapper, ShapeMismatchError


class TestModelServingWrapper(TestCase):

    TESTBASE_PATH = "sqlite:///base.sqlite"

    @pytest.fixture(autouse=True)
    def temp_database(self, tmpdir):
        tmpdir.chdir()
        engine = create_engine(self.TESTBASE_PATH)
        Base.metadata.create_all(engine)

    def get_session(self):
        engine = create_engine(self.TESTBASE_PATH)
        Session = sessionmaker(bind=engine)
        session = Session()
        return session

    def test__fit(self):
        session = self.get_session()

        session.add(DatasetSample(image=pickle.dumps(np.array([0])), label=0))
        session.add(DatasetSample(image=pickle.dumps(np.array([1])), label=1))

        session.commit()

        t = ModelServingWrapper(LinearRegression(), self.TESTBASE_PATH)
        t._fit()

        self.assertAlmostEqual(t._model.coef_[0], 1)
        self.assertAlmostEqual(t._model.intercept_, 0)

    def test_predict(self):
        session = self.get_session()

        session.add(DatasetSample(image=pickle.dumps(np.array([0])), label=0))
        session.add(DatasetSample(image=pickle.dumps(np.array([1])), label=1))

        session.commit()

        t = ModelServingWrapper(LogisticRegression(), self.TESTBASE_PATH)
        t.fit()

        self.assertEqual(t.predict(np.array([1])), 1)

    def test_predict_incorrect_shape(self):
        session = self.get_session()

        session.add(DatasetSample(image=pickle.dumps(np.array([0, 2])), label=1))
        session.add(DatasetSample(image=pickle.dumps(np.array([0, 1])), label=0))
        session.commit()

        t = ModelServingWrapper(LogisticRegression(), self.TESTBASE_PATH)
        t.fit()

        with self.assertRaises(ShapeMismatchError):
            t.predict(np.array([0]))

    def test_predict_not_fitted(self):
        t = ModelServingWrapper(LogisticRegression(), self.TESTBASE_PATH)

        with self.assertRaises(ValueError):
            t.predict(np.array([0, 1, 2]))

    def test__init__with_fitted_model(self):
        fitted_model = LogisticRegression()
        fitted_model.fit(np.array([[0, 1, 2], [4, 5, 6]]), np.array([0, 1]))

        with self.assertRaises(ValueError):
            ModelServingWrapper(fitted_model, self.TESTBASE_PATH)

    def test__get_training_dataset_empty(self):

        t = ModelServingWrapper(LinearRegression(), self.TESTBASE_PATH)

        with self.assertRaises(ValueError):
            t._get_training_dataset()

    def test__get_training_dataset_one_sample(self):
        session = self.get_session()

        test_vector = np.random.random(100)
        test_label = np.random.randint(0, 128)

        session.add(DatasetSample(image=pickle.dumps(test_vector), label=test_label))
        session.commit()

        t = ModelServingWrapper(LinearRegression(), self.TESTBASE_PATH)
        images, labels, shape = t._get_training_dataset()

        self.assertTrue(np.array_equal(images, np.vstack([test_vector])))
        self.assertTrue(np.array_equal(labels, np.array([test_label])))
        self.assertEqual(shape, test_vector.shape)

    def test__get_training_dataset_shapes_mismatch(self):
        session = self.get_session()

        session.add(DatasetSample(image=pickle.dumps(np.array([0])), label=0))
        session.add(DatasetSample(image=pickle.dumps(np.array([0, 1])), label=0))
        session.commit()

        t = ModelServingWrapper(LinearRegression(), self.TESTBASE_PATH)

        with self.assertRaises(ValueError):
            images, labels, shape = t._get_training_dataset()

