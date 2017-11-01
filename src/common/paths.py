import os

JPEG_EXT = '.jpg'
DATA_ROOT = 'data/data/'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
#TRAIN_TF_RECORDS = os.path.join(ROOT, 'dogs_train.tfrecords')
TRAIN_TF_RECORDS = os.path.join(DATA_ROOT, 'stanford.tfrecords')
TEST_TF_RECORDS = os.path.join(DATA_ROOT, 'dogs_test.tfrecords')
LABELS = os.path.join(DATA_ROOT, 'train', 'labels.csv')
BREEDS = os.path.join(DATA_ROOT, 'breeds.csv')
IMAGENET_GRAPH_DEF = 'data/frozen/inception/classify_image_graph_def.pb'
TEST_PREDICTIONS = 'data/predictions.csv'
METRICS_DIR = 'data/metrics'
TRAIN_CONFUSION = os.path.join(METRICS_DIR, 'training_confusion.csv')
FROZEN_MODELS_DIR = 'data/frozen'
CHECKPOINTS_DIR = 'data/checkpoints'
GRAPHS_DIR = 'data/graphs'
SUMMARY_DIR = 'data/summary'
STANFORD_DS_DIR = os.path.join(DATA_ROOT, 'stanford_ds')
STANFORD_DS_TF_RECORDS = os.path.join(DATA_ROOT, 'stanford.tfrecords')

