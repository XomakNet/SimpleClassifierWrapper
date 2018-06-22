#!/usr/bin/env python3


"""
This script is used to fit the LogisticRegression model, using ModelServingWrapper and
dataset, located in the database.

Example:

    learner.py mssql+pyodbc://user:pass@127.0.0.1/mnist?driver=ODBC+Driver+17+for+SQL+Server models/model.pickle

"""

import argparse
import pickle

from sklearn.linear_model import LogisticRegression

from app.wrapper import ModelServingWrapper

parser = argparse.ArgumentParser(description='Fits the LogisticRegression model, using samples from given '
                                             'database and saves to file')

parser.add_argument('database_path', type=str, help='SQLAlchemy-compatible Database URL')
parser.add_argument('file_path', type=str, help='Path to file with model')

args = parser.parse_args()

model = ModelServingWrapper(LogisticRegression(), args.database_path)
accuracy = model.fit()

with open(args.file_path, 'wb') as handle:
    pickle.dump(model, handle)
