import yaml
from dictionaries import Dict
import argparse


def parse_config_file(config_filename):
    with open(config_filename, 'r') as yml_file:
        cfg = yaml.load(yml_file)

    arg_config_ = Dict(copy=True, name='config')
    arg_config_.MODEL_NAME = str(cfg["MODEL"]["MODEL_NAME"])
    arg_config_.PRETAIN_MODEL = str(cfg["MODEL"]["PRETRAIN_MODEL"])
    arg_config_.PRETAIN_MODEL_PATH = str(cfg["MODEL"]["PRETAIN_MODEL_PATH"])
    arg_config_.PRETAIN_MODEL_URL = str(cfg["MODEL"]["PRETAIN_MODEL_URL"])
    arg_config_.EXCLUDE_NODES = cfg["MODEL"]["EXCLUDE_NODES"]
    arg_config_.INPUT_HEIGHT = int(cfg["MODEL"]["INPUT_HEIGHT"])
    arg_config_.INPUT_WIDTH = int(cfg["MODEL"]["INPUT_WIDTH"])
    arg_config_.CATEGORIES = int(cfg["MODEL"]["CLASSES"])

    arg_config_.TRAIN_BATCH_SIZE = int(cfg["TRAIN"]["BATCH_SIZE"])
    arg_config_.TRAIN_EPOCHS_COUNT = int(cfg["TRAIN"]["EPOCHS_COUNT"])
    arg_config_.TRAIN_LEARNING_RATE = float(cfg["TRAIN"]["LEARNING_RATE"])
    arg_config_.TRAIN_KEEP_PROB = float(cfg['TRAIN']['KEEP_PROB'])
    arg_config_.TRAIN_TF_RECORDS = str(cfg["TRAIN"]["TF_RECORDS_PATH"])
    arg_config_.TRAIN_EPOCHS_BEFORE_DECAY = float(cfg["TRAIN"]["TRAIN_EPOCHS_BEFORE_DECAY"])
    arg_config_.TRAIN_RATE_DECAY_FACTOR = float(cfg["TRAIN"]["TRAIN_RATE_DECAY_FACTOR"])
    arg_config_.TRAINABLE_SCOPES = cfg["TRAIN"]["TRAINABLE_SCOPES"]
    arg_config_.L2_WEIGHT_DECAY = float(cfg["TRAIN"]["L2_WEIGHT_DECAY"])

    arg_config_.EVAL_BATCH_SIZE = cfg["EVAL"]["BATCH_SIZE"]
    arg_config_.EVAL_TF_RECORDS = str(cfg["EVAL"]["TF_RECORDS"])
    arg_config_.EVAL_OUTPUT = str(cfg["EVAL"]["OUTPUT_PATH"])

    arg_config_.TEST_BATCH_SIZE = cfg["TEST"]["BATCH_SIZE"]
    arg_config_.TEST_TF_RECORDS = str(cfg["TEST"]["TF_RECORDS"])
    arg_config_.TEST_OUTPUT = str(cfg["TEST"]["OUTPUT_PATH"])

    return arg_config_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c',
                        dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    arg_config = parse_config_file(args.config_filename)
    print(type(arg_config.EXCLUDE_NODES))
    print(arg_config.TRAINABLE_SCOPES)
