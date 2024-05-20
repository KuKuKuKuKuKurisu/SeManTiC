import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
# from peft import LoraConfig, get_peft_model, TaskType
from .recommend_module import Encoder, DSI

from config import GlobalConfig, RecommendTrainConfig
if RecommendTrainConfig.simmc2:
    from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
else:
    from config.dataset_config import DatasetConfig
from config.constants import *
from utils.mask import *
from utils import load_pkl

class DST_AWARE(nn.Module):
    def __init__(self, clip_enc_model, state_keys):
        super(DST_AWARE, self).__init__()

        # constant
        self.text_emb_size = 512
        self.layer_num = CROSS_LAYER_NUM
        self.head_num = CROSS_ATTN_HEAD_NUM

        self.img_emb_size = 512

        self.co_layer_num = CO_CROSS_LAYER_NUM

        self.co_emb_size = TEXT_EMB_SIZE + self.img_emb_size

        self.text_d_k = int(TEXT_EMB_SIZE/CROSS_ATTN_HEAD_NUM)

        self.padding_id = PADDING_ID

        # embedding layer
        self.encoder = clip_enc_model
        # lora_config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,
        #     target_modules=["c_fc", "c_proj", "out_proj"],
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type=TaskType.CAUSAL_LM
        # )
        # self.encoder = get_peft_model(self.encoder, lora_config).to(dtype=torch.bfloat16)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # state_keys
        self.state_keys = state_keys.to(GlobalConfig.device)

        # image_encoder
        self.image_encoder = nn.Linear(self.img_emb_size, self.text_emb_size)

        # context self_attention
        self.context_text_self_attention = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                                   d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        self.context_image_self_attention = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                                    d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        # MSE
        self.mse = nn.MSELoss(reduction='mean')

        # kl
        self.kl = nn.KLDivLoss(reduction="none")

        # tanh
        self.tanh = nn.Tanh()

        # softmax
        self.softmax = nn.Softmax(dim = -1)

        # relu
        self.relu = nn.ReLU(inplace=False)

        # leaky relu
        self.leaky_relu = nn.LeakyReLU(inplace=False)

    def get_text_emb(self, text_ids, type):
        text_features = self.encoder.encode_text(text_ids).float()
        return text_features

    def get_image_emb(self, image):
        image_emb = self.encoder.encode_image(image).float()
        return image_emb

    def get_context_emb(self, text_ids, images, image_mask):

        # context text emb
        sent_emb = self.get_text_emb(text_ids, 'history').unsqueeze(1)

        # context image emb
        image_emb = []
        for img_id in range(len(images[0])):
            # image emb
            temp_image_emb = self.get_image_emb(images[:, img_id, :, :, :]).unsqueeze(1)

            non_pad_mask = get_non_pad_mask(image_mask[:, img_id:img_id+1], self.padding_id)
            temp_image_emb *= non_pad_mask
            image_emb.append(temp_image_emb)

        image_emb = torch.cat(image_emb, dim = 1)
        image_emb = self.leaky_relu(torch.sum(image_emb, dim = 1)).unsqueeze(1)

        return sent_emb, image_emb

    def turn_level_self_attn(self, context_txt_emb, context_img_emb, turn_mask):

        attn_mask = get_attn_key_pad_mask(seq_k=turn_mask, seq_q=turn_mask, padding_id=self.padding_id)
        non_pad_mask = get_non_pad_mask(turn_mask, self.padding_id)

        context_txt_emb, = self.context_text_self_attention(context_txt_emb, context_txt_emb, non_pad_mask, attn_mask)
        context_img_emb, = self.context_image_self_attention(context_img_emb, context_txt_emb, non_pad_mask, attn_mask)

        return context_txt_emb[:,-1,:], context_img_emb[:,-1,:]

    def get_item_embedding(self, attrs_id, image):
        # text emb
        text_embedding = self.get_text_emb(attrs_id, 'attrs')

        # image emb
        image_embedding = self.get_image_emb(image)

        return text_embedding, image_embedding

    def get_kl_distribution(self, query_emb, key_emb):
        return self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(key_emb[0]), 1), key_emb, dim = -1))

    def cal_kl_loss(self, input, target):
        return torch.sum(self.kl(input, target), dim = -1)

    def get_kl_loss(self, target_txt_kl, target_img_kl, input_txt_kl, input_img_kl):

        text_klloss = 0.5 * self.cal_kl_loss(torch.log(target_txt_kl), input_txt_kl) + \
                      0.5 * self.cal_kl_loss(torch.log(input_txt_kl), target_txt_kl)
        image_klloss = 0.5 * self.cal_kl_loss(torch.log(target_img_kl), input_img_kl) + \
                       0.5 * self.cal_kl_loss(torch.log(input_img_kl), target_img_kl)

        kl_loss = text_klloss + image_klloss
        return kl_loss

    def get_learn_loss(self, input, target):
        return self.mse(input, target)

    def forward(self, context_dialog, pos_products, neg_products, eval = False):

        texts, images, image_mask, utter_type, dst = context_dialog
        pos_image_num, pos_images, pos_product_texts = pos_products
        neg_image_num, neg_images, neg_product_texts = neg_products

        # to device
        text_ids = texts['input_ids'].to(GlobalConfig.device)
        text_type = texts['text_type'].to(GlobalConfig.device)
        turn_mask = texts['turn_mask'].to(GlobalConfig.device)

        pos_attrs_id = pos_product_texts['input_ids'].to(GlobalConfig.device)
        pos_attrs_type = pos_product_texts['text_type'].to(GlobalConfig.device)

        neg_attrs_id = neg_product_texts['input_ids'].to(GlobalConfig.device)
        neg_attrs_type = neg_product_texts['text_type'].to(GlobalConfig.device)

        images = images.to(GlobalConfig.device)

        image_mask = image_mask.to(GlobalConfig.device)

        pos_imgs_num = pos_image_num.to(GlobalConfig.device)
        neg_imgs_num = neg_image_num.to(GlobalConfig.device)

        pos_imgs = pos_images.to(GlobalConfig.device)
        neg_imgs = neg_images.to(GlobalConfig.device)

        # constants
        batch_size = len(images)
        ones = torch.ones(batch_size).to(GlobalConfig.device)
        zeros = torch.zeros(batch_size).to(GlobalConfig.device)

        # context emb
        context_text_embeddings = []
        context_image_embeddings = []

        for turn_id in range(DatasetConfig.dialog_context_size):
            context_text_embedding, context_image_embedding = self.get_context_emb(text_ids[:, turn_id, :], images[:, turn_id, :, :, :, :], image_mask[:, turn_id, :])
            context_text_embeddings.append(context_text_embedding)
            context_image_embeddings.append(context_image_embedding)

        # dst or query aware
        context_txt_emb, context_img_emb = self.turn_level_self_attn(torch.cat(context_text_embeddings, dim = 1), torch.cat(context_image_embeddings, dim = 1), turn_mask)

        # sims
        neg_sims = []
        neg_text_image_sims = []
        neg_kl_losses = []

        # neg items
        for neg_id in range(DatasetConfig.max_neg_num):

            neg_text_embedding, neg_image_embedding = self.get_item_embedding(neg_attrs_id[:, neg_id, :],neg_imgs[:, neg_id, :, :, :])
            # neg sim
            pred_neg_text_sim = cosine_similarity(context_txt_emb, neg_text_embedding, dim = -1)
            pred_neg_image_sim = cosine_similarity(context_img_emb, neg_image_embedding, dim = -1)
            neg_sims.append(self.tanh(pred_neg_text_sim + pred_neg_image_sim))

        # loss component
        mask = get_mask(DatasetConfig.max_neg_num, neg_imgs_num, GlobalConfig.device)
        losses = [[] for _ in range(DatasetConfig.max_pos_num)]
        pred_losses = [[] for _ in range(DatasetConfig.max_pos_num)]
        kl_losses = [[] for _ in range(DatasetConfig.max_pos_num)]

        pos_mask = get_mask(DatasetConfig.max_pos_num, pos_imgs_num, GlobalConfig.device)

        pos_text_image_sims = []

        # eval component
        rank_temp = torch.zeros(DatasetConfig.max_pos_num, batch_size, dtype=torch.long).to(GlobalConfig.device)

        for pos_id in range(DatasetConfig.max_pos_num):

            pos_text_embedding, pos_image_embedding = self.get_item_embedding(pos_attrs_id[:, pos_id, :], pos_imgs[:, pos_id, :, :, :])
            # pos sim
            pred_pos_text_sim = cosine_similarity(context_txt_emb, pos_text_embedding, dim = -1)
            pred_pos_image_sim = cosine_similarity(context_img_emb, pos_image_embedding, dim = -1)
            pos_sim = self.tanh(pred_pos_text_sim + pred_pos_image_sim)

            for neg_id in range(len(neg_sims)):
                # sim loss
                loss = torch.max(zeros, ones - pos_sim + neg_sims[neg_id])
                rank_temp[pos_id] += torch.lt(pos_sim, neg_sims[neg_id]).long() * mask[:, neg_id]
                losses[pos_id].append(loss)

            # sim loss
            losses[pos_id] = torch.stack(losses[pos_id])
            losses[pos_id] = losses[pos_id].transpose(0, 1)
            losses[pos_id] = losses[pos_id].masked_select(mask.bool()).mean()

        loss = 0
        count = 0
        for loss_tmp in losses:
            if loss_tmp != []:
                count += 1
                loss += loss_tmp
        loss /= count

        sim_loss = float(loss)

        if eval:
            return loss, rank_temp, pos_imgs_num
        else:
            return loss, (sim_loss, 0, 0, 0, 0, 0)