"""Tidy data module."""
import copy
from os.path import isfile, join
from utils import save_pkl, load_pkl, process_txt
from config import DatasetConfig, RecommendTrainConfig
from tqdm import tqdm
from utils import save_pkl, get_product_path
from dataset.model import TidyUtterance, Utterance

from transformers import BertTokenizer
from config.constants import *

import json
def get_images_path():
    """Get images (URL and filenames of local images mapping).

    URL -> Path => URL -> index & index -> Path

    Returns:
        Dict[str, int]: Image URL to index.
        List[str]: Index to the filename of the local image.

    """
    # Get URL to filename mapping dict.
    with open(DatasetConfig.url2img, 'r') as file:
        url_image_pairs = [line.strip().split(' ') for line in file.readlines()]
    url_image_pairs = [(p[0], p[1]) for p in url_image_pairs]
    url2img = dict(url_image_pairs)

    # Divided it into two steps.
    # URL -> Path => URL -> index & index -> Pathtrain
    # Element of index 0 should be empty image.
    image_url_id = {'': 0}
    image_paths = ['']
    for url, img in url2img.items():
        image_url_id[url] = len(image_url_id)
        image_paths.append(img)

    return image_url_id, image_paths

def generate_tidy_data_file(raw_data ,image_paths):
    """Generate tidy data file.

    Args:
        raw_data (RawData): Raw data.
        mode (str): A single mode.

    """

    # Get raw data dialogs according to its mode.
    dialogs = raw_data
    assert dialogs is not None

    # tokenizer
    with open(DatasetConfig.special_tokens, "rb") as handle:
        special_tokens_dict = json.load(handle)

    tokenizer = BertTokenizer.from_pretrained(PRETRAIN_TEXT_ENCODER)
    tokenizer.add_special_tokens(special_tokens_dict)

    # dialog = get_recommend_task_items(raw_data.image_paths, dialogs[0])

    tidy_dialogs = []
    valid_pos_num = []
    valid_neg_num = []
    for item_idx, dialog in enumerate(tqdm(dialogs)):
        # Get items according to different TASKS.
        if dialog == []:
            continue
        cleaned_dialogs, pos_img_num, neg_img_num = get_recommend_task_items(image_paths, dialog, tokenizer)
        tidy_dialogs.extend(cleaned_dialogs)
        valid_pos_num.extend(pos_img_num)
        valid_neg_num.extend(neg_img_num)

    print('valid pos num: ', sum(valid_pos_num)/len(valid_pos_num))
    print('valid neg num: ', sum(valid_neg_num)/len(valid_neg_num))

def get_init_pad_utters(tokenizer):
    """Get initial padding utterances.

    Returns:
        List[TidyUtterance]

    """
    EMPTY_TEXT = tokenizer([''], padding='max_length', max_length = DatasetConfig.product_text_max_len, truncation=True, return_tensors='pt',
                           add_special_tokens=DatasetConfig.add_special_tokens)
    EMPTY_TEXT['input_ids'] = EMPTY_TEXT['input_ids'].tolist()
    EMPTY_TEXT['attention_mask'] = EMPTY_TEXT['attention_mask'].tolist()
    del EMPTY_TEXT['token_type_ids']

    utters = []

    for i in range(DatasetConfig.dialog_context_size):
        if i % 2 == 0:
            speaker = 'user'
        else:
            speaker = 'system'

        utter = Utterance(speaker, 'pad', EMPTY_TEXT, [], [], EMPTY_TEXT)
        utters.append(TidyUtterance(utter))
    return utters

def get_valid_image(image_paths, images):
    """ Get valid images in `images`.

    Args:
        image_paths (List[str]): Image paths.
        images (List[int]): Images.

    Returns:
        valid_images (List[int]): Valid images.

    """

    res_images = []
    for image_id in images:
        image_path = join(DatasetConfig.image_data_directory,
                          image_paths[image_id])
        if not isfile(image_path):
            continue
        product_path = get_product_path(image_paths[image_id])
        if not isfile(product_path):
            continue
        res_images.append(image_id)
    return res_images


def get_recommend_task_items(image_paths, dialog, tokenizer):
    """Get items for recommend task from a single dialog.

    Args:
        image_paths (List[str]): Image paths.
        dialog (Dialog): Dialog.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.

    """

    dialogs = []
    utterances = get_init_pad_utters(tokenizer)
    context_size = DatasetConfig.dialog_context_size

    pos_img_num = []
    neg_img_num = []

    # Merge
    tmp_dialogs = []
    tmp_utters = []
    for utter in dialog:
        if utter.text is None:
            utter.text = []
        if utter.pos_images is None:
            utter.pos_images = []
        if utter.neg_images is None:
            utter.neg_images = []

        if len(tmp_utters) == 0:
            tmp_utters.append(Utterance(utter.speaker, utter.utter_type, utter.text, utter.pos_images, utter.neg_images, utter.dst))
            continue
        if utter.speaker != tmp_utters[-1].speaker:
            tmp_dialogs.append(tmp_utters[-1])
            tmp_utters.append(Utterance(utter.speaker, utter.utter_type, utter.text, utter.pos_images, utter.neg_images, utter.dst))
        else:
            last_utter = tmp_utters[-1]
            last_utter.text = last_utter.text + ' ' + utter.text
            last_utter.pos_images.extend(utter.pos_images)
            last_utter.neg_images.extend(utter.neg_images)
            last_utter.utter_type = utter.utter_type
            last_utter.dst = utter.dst
            tmp_utters[-1] = Utterance(last_utter.speaker, last_utter.utter_type, last_utter.text, last_utter.pos_images, last_utter.neg_images, last_utter.dst)
    tmp_dialogs.append(tmp_utters[-1])

    recommendations = 0

    for utter in tmp_dialogs:

        utterances.append(TidyUtterance(utter))

        if utter.speaker == 'system' and utter.pos_images and utter.neg_images:
            # pos_img_num.append(len(utter.pos_images))
            # neg_img_num.append(len(utter.neg_images))
            recommendations+=1
            utterances = utterances[-(context_size + 1):]
            dialogs.append(copy.copy(utterances))

        # if utter.speaker == 'system' and utter.pos_images and utter.neg_images:
        #     utter.dst = dst_format
        #     utterances.append(TidyUtterance(utter))
        #
        #     pos_img_num.append(len(utter.pos_images))
        #     neg_img_num.append(len(utter.neg_images))
        #
        #     utterances = utterances[-(context_size + 1):]
        #     dialogs.append(copy.copy(utterances))
        # else:
        #     utter.dst = None
        #     utterances.append(TidyUtterance(utter))
    pos_img_num.append(recommendations)
    return dialogs, pos_img_num, neg_img_num

if __name__ == '__main__':
    splits = ['train', 'valid', 'test', 'full_valid', 'full_test']
    image_url_id, image_paths = get_images_path()
    for split in splits:
        if split == 'train':
            raw_output_path = DatasetConfig.train_raw_data_file
        elif split == 'valid':
            raw_output_path = DatasetConfig.valid_raw_data_file
        elif split == 'test':
            raw_output_path = DatasetConfig.test_raw_data_file
        elif split == 'full_valid':
            raw_output_path = DatasetConfig.wodst_valid_raw_data_file
        else:
            raw_output_path = DatasetConfig.wodst_test_raw_data_file
        print(split)
        dialogs = load_pkl(raw_output_path)
        generate_tidy_data_file(dialogs, image_paths)