from sqlalchemy import Column, Integer, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DatasetSample(Base):
    """
    Describes the dataset sample
    """
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True)
    image = Column(LargeBinary)
    label = Column(Integer)
