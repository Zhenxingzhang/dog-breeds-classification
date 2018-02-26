import os

JPEG_EXT = '.jpg'
DATA_ROOT = '/data/dog_breeds/stanford_ds'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TRAIN_CSV_FILE = os.path.join(TRAIN_DIR, 'train.csv')
VAL_CSV_FILE = os.path.join(VAL_DIR, 'val.csv')

TRAIN_TF_RECORDS = os.path.join(TRAIN_DIR, 'train.tfrecord')
VAL_TF_RECORDS = os.path.join(VAL_DIR, 'val.tfrecord')

TRAIN_SUMMARY_DIR = "/data/summary/dog_breeds/train"
VAL_SUMMARY_DIR = "/data/summary/dog_breeds/val"

CHECKPOINT_DIR = '/data/checkpoints/dog_breeds'

STANFORD_DATA_DIR = '/data/dog_breeds/stanford_ds'
STANFORD_CSV_FILE = os.path.join(STANFORD_DATA_DIR, 'stanford.csv')
STANFORD_TFRECORD = os.path.join(STANFORD_DATA_DIR, 'stanford.tfrecord')

# STANFORD_TRAIN_CSV = os.path.join(STANFORD_DATA_DIR, 'train.csv')

KG_DATA_DIR = "/data/dog_breeds/kaggle"

KG_CSV = os.path.join(KG_DATA_DIR, 'labels.csv')
KG_TRAIN_CSV = os.path.join(KG_DATA_DIR, 'train.csv')
KG_TRAIN_TFRECORD = os.path.join(KG_DATA_DIR, 'train.tfrecord')

TEST_CSV_RECORDS = os.path.join(KG_DATA_DIR, 'dogs_test.csv')
TEST_TF_RECORDS = os.path.join(KG_DATA_DIR, 'dogs_test.tfrecords')


# LABELS = os.path.join(DATA_ROOT, 'train', 'labels.csv')
# IMAGENET_GRAPH_DEF = '/data/frozen/inception/classify_image_graph_def.pb'
# TEST_PREDICTIONS = '/data/outputs/dog_breeds/predictions.csv'
# METRICS_DIR = '/data/metrics/dog_breeds'
# TRAIN_CONFUSION = os.path.join(METRICS_DIR, 'training_confusion.csv')
# FROZEN_MODELS_DIR = '/data/frozen/dog_breeds'
# GRAPHS_DIR = '/data/graphs/dog_breeds'
# SUMMARY_DIR = '/data/summary/dog_breeds'
# STANFORD_DS_DIR = os.path.join(DATA_ROOT, 'stanford_ds')
# STANFORD_DS_TF_RECORDS = os.path.join(DATA_ROOT, 'stanford_ds', 'train', 'stanford.tfrecord')
