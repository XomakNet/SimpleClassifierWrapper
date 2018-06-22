import json
import os

import pytest

import webapp as webapp

API_ENDPOINT = '/api/classify'
RESOURCES_DIR = 'tests/resources/'


@pytest.fixture
def client():
    webapp.app.config['TESTING'] = True
    client = webapp.app.test_client()
    yield client


def test_classify_empty(client):
    response = client.post(API_ENDPOINT)
    assert response.status_code == 400


def test_classify_valid(client):
    file_to_upload = open(os.path.join(RESOURCES_DIR, 'mnist_test_zero.png'), 'rb')
    response = client.post(API_ENDPOINT, data={'image': (file_to_upload, 'image.png')})

    json_data = json.loads(response.data)

    assert 'class' in json_data
    assert response.status_code == 200


def test_classify_not_image(client):
    file_to_upload = open(os.path.join(RESOURCES_DIR, 'not_an_image.txt'), 'rb')
    response = client.post(API_ENDPOINT, data={'image': (file_to_upload, 'image.png')})
    assert response.status_code == 400
    assert "The given file is not an image" in response.data.decode("utf-8")


def test_classify_shape_mismatch(client):
    file_to_upload = open(os.path.join(RESOURCES_DIR, 'not_valid_shape.png'), 'rb')
    response = client.post(API_ENDPOINT, data={'image': (file_to_upload, 'image.png')})
    assert response.status_code == 400
