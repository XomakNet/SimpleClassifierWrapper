#!/usr/bin/env python3

"""
Creates table in the database (if it was not created) and imports all samples from the given files
(samples and labels).

Example:

    data_importer.py train-images.idx3-ubyte train-labels.idx1-ubyte
    mssql+pyodbc://user:pass@127.0.0.1/mnist?driver=ODBC+Driver+17+for+SQL+Server

"""

import argparse
import pickle

import idx2numpy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, DatasetSample

parser = argparse.ArgumentParser(description='Imports data from IDX to the database with given path')
parser.add_argument('training_images_path', type=str,
                    help='Path to the IDX file with training images')
parser.add_argument('training_labels_path', type=str,
                    help='Path to the IDX file with training labels')
parser.add_argument('database_path', type=str,
                    help='SQLAlchemy-compatible Database URL')

args = parser.parse_args()

engine = create_engine(args.database_path)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

images = idx2numpy.convert_from_file(args.training_images_path)
labels = idx2numpy.convert_from_file(args.training_labels_path)

for image, label in zip(images, labels):
    item = DatasetSample(image=pickle.dumps(image), label=int(label))
    session.add(item)

session.commit()
