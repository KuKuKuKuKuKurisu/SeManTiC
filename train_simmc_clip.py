#!/usr/bin/python3

from os.path import isfile, join
import json
import torch

from config import RecommendTrainConfig, GlobalConfig
from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
from dataset.dataset_clip import Dataset
from lib.recommend_train import recommend_train
from utils import load_pkl
from collections import namedtuple
from transformers import BertTokenizer, BertModel
from config.constants import *
import torch.nn as nn
import argparse
from widget.dst_aware_model_clip import DST_AWARE
from torchvision.models import resnet18

CommonData = namedtuple('CommonData',
                        ['image_paths'])
from clip import clip
def train(args):
    """Train model.

    Args:
        task (int): Task.
        model_file_name (str): Model file name (saved or to be saved).

    """

    # state keys
    state_keys = clip.tokenize(SIMMC_SLOT_TOKENS, context_length = 1, truncate = True)
    state_keys = state_keys.squeeze()

    # common data
    # Check if data exists.
    if not isfile(DatasetConfig.common_raw_data_file):
        raise ValueError('No common raw data.')
    # Load extracted common data.
    common_data = load_pkl(DatasetConfig.common_raw_data_file)

    # dialogue data
    train_dialog_data_file = DatasetConfig.recommend_train_dialog_file
    valid_dialog_data_file = DatasetConfig.recommend_valid_dialog_file
    test_dialog_data_file = DatasetConfig.recommend_test_dialog_file

    if not isfile(train_dialog_data_file):
        raise ValueError('No train dialog data file.')
    if not isfile(valid_dialog_data_file):
        raise ValueError('No valid dialog data file.')
    if not isfile(test_dialog_data_file):
        raise ValueError('No test dialog data file.')

    # Load extracted dialogs.
    train_dialogs = load_pkl(train_dialog_data_file)
    valid_dialogs = load_pkl(valid_dialog_data_file)
    test_dialogs = load_pkl(test_dialog_data_file)

    enc_model, preprocess = clip.load("ViT-B/32")

    # Dataset wrap.
    train_dataset = Dataset(
        'train',
        common_data.image_paths,
        train_dialogs,
        preprocess)
    valid_dataset = Dataset(
        'valid',
        common_data.image_paths,
        valid_dialogs,
        preprocess)
    test_dataset = Dataset(
        'test',
        common_data.image_paths,
        test_dialogs,
        preprocess)

    print('Train dataset size:', len(train_dataset))
    print('Valid dataset size:', len(valid_dataset))
    print('Test dataset size:', len(test_dataset))

    fewshot_dataset = None

    # init model.
    encoder = DST_AWARE(enc_model, state_keys)
    model_file = join(DatasetConfig.dump_dir, args['model_file_name'])

    recommend_train(
        encoder,
        train_dataset,
        valid_dataset,
        test_dataset,
        model_file,
        fewshot_dataset
    )

def parse_cmd():
    """Parse commandline parameters.

    Returns:
        Dict[str, List[str]]: Parse result.

    """

    # Definition of argument parser.
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('model_file_name', metavar='<model_file_name>')
    parser.add_argument('device', metavar='<device>')

    # Namespace -> Dict
    parse_res = vars(parser.parse_args())
    return parse_res

def main():
    # Parse commandline parameters and standardize.
    parse_result = parse_cmd()
    GlobalConfig.device = torch.device(parse_result['device'] if torch.cuda.is_available() else "cpu")
    train(parse_result)

if __name__ == '__main__':
    main()
