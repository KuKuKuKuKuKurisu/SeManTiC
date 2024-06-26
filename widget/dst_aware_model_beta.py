import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.functional import cosine_similarity

from .recommend_module import Encoder, DSI_beta

from config import GlobalConfig, RecommendTrainConfig
if RecommendTrainConfig.simmc2:
    from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
else:
    from config.dataset_config import DatasetConfig
from config.constants import *
from utils.mask import *
from utils import load_pkl

class DST_AWARE(nn.Module):
    def __init__(self, pretrained_embedding, pretrained_image_encoder, state_keys):
        super(DST_AWARE, self).__init__()

        # constant
        self.text_emb_size = TEXT_EMB_SIZE
        self.layer_num = CROSS_LAYER_NUM
        self.head_num = CROSS_ATTN_HEAD_NUM

        self.img_emb_size = 512

        self.co_layer_num = CO_CROSS_LAYER_NUM

        self.co_emb_size = TEXT_EMB_SIZE + self.img_emb_size

        self.text_d_k = int(TEXT_EMB_SIZE/CROSS_ATTN_HEAD_NUM)

        self.padding_id = PADDING_ID

        self.temperature = 3
        self.dsi_layer_num = HOPS

        # embedding layer
        self.word_embeddings = pretrained_embedding.word_embeddings
        self.position_embeddings = pretrained_embedding.position_embeddings

        # pretrained_image_encoder
        self.pretrained_image_encoder = pretrained_image_encoder
        for param in self.pretrained_image_encoder.parameters():
            param.requires_grad = False

        # state_keys
        self.state_keys = state_keys.to(GlobalConfig.device)

        # text_encoder
        self.txt_encoder = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                   d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        # self.attrs_encoder = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
        #                              d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        # image_encoder
        self.image_encoder = nn.Linear(self.img_emb_size, self.text_emb_size)
        # self.image_encoder = nn.Sequential(
        #     nn.Linear(self.img_emb_size, 256),
        #     nn.GELU(),
        #     nn.Linear(256, self.text_emb_size)
        # )

        # state aware emcoder
        self.get_state_aware = DSI_beta(n_hop=self.dsi_layer_num, d_model=self.text_emb_size, temperature = self.temperature)

        # pred state aware emcoder
        self.state_value_pred = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                        d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        # context self_attention
        self.context_text_self_attention = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                                   d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        self.context_image_self_attention = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                                    d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        # MSE
        self.mse = nn.MSELoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean')

        # kl
        self.kl = nn.KLDivLoss(reduction="none")

        # ce
        self.ce = nn.CrossEntropyLoss(reduction='none')

        # tanh
        self.tanh = nn.Tanh()

        # softmax
        self.softmax = nn.Softmax(dim = -1)

        # relu
        self.relu = nn.ReLU()

        # leaky relu
        self.leaky_relu = nn.GELU()

    def get_text_emb(self, text_ids, type):

        word_emb = self.word_embeddings(text_ids)

        if type == 'history':
            pos_emb = self.position_embeddings(torch.tensor([pos for pos in range(len(text_ids[0]))]).repeat(len(text_ids), 1).to(GlobalConfig.device))
            word_emb = word_emb + pos_emb

        attn_mask = get_attn_key_pad_mask(seq_k=text_ids, seq_q=text_ids, padding_id=self.padding_id)
        non_pad_mask = get_non_pad_mask(text_ids, self.padding_id)
        word_emb, = self.txt_encoder(word_emb, word_emb, non_pad_mask, attn_mask)

        # if type == 'history':
        #     attn_mask = get_attn_key_pad_mask(seq_k=text_ids, seq_q=text_ids, padding_id=self.padding_id)
        #     non_pad_mask = get_non_pad_mask(text_ids, self.padding_id)
        #     word_emb, = self.history_encoder(word_emb, word_emb, non_pad_mask, attn_mask)
        # elif type == 'attrs':
        #     attn_mask = get_attn_key_pad_mask(seq_k=text_ids, seq_q=text_ids, padding_id=self.padding_id)
        #     non_pad_mask = get_non_pad_mask(text_ids, self.padding_id)
        #     word_emb, = self.attrs_encoder(word_emb, word_emb, non_pad_mask, attn_mask)

        return word_emb

    def get_image_emb(self, image):
        image_emb = self.pretrained_image_encoder(image).view(len(image), self.img_emb_size)
        image_emb = self.image_encoder(image_emb)
        return image_emb

    def get_state_emb(self, dst_ids, dst_attention_mask, batch_size):

        state_key_emb = self.word_embeddings(self.state_keys.unsqueeze(0)).repeat(batch_size, 1, 1)

        state_emb = self.word_embeddings(dst_ids)

        state_value_emb = []
        for key_id in range(len(state_key_emb[0])):
            unmask_num = torch.sum(dst_attention_mask[:, key_id * DatasetConfig.state_value_max_len : (key_id + 1) * DatasetConfig.state_value_max_len], dim = -1).view(-1, 1, 1)
            value_emb = state_emb[:, key_id * DatasetConfig.state_value_max_len : (key_id + 1) * DatasetConfig.state_value_max_len, :]
            value_emb = torch.sum(value_emb, dim = 1, keepdim=True)
            value_emb = value_emb/unmask_num
            state_value_emb.append(value_emb)

        state_value_emb = torch.cat(state_value_emb, dim = 1)

        return self.leaky_relu(state_value_emb)

    def get_context_emb(self, text_ids, images, image_mask, turn_id):

        # context text emb
        sent_emb = self.get_text_emb(text_ids, 'history')
        sent_emb = self.leaky_relu(torch.mean(sent_emb, dim = 1)).unsqueeze(1)

        # context image emb
        image_emb = []
        for img_id in range(len(images[0])):
            # image emb
            temp_image_emb = self.get_image_emb(images[:, img_id, :, :, :])
            temp_image_emb = temp_image_emb.unsqueeze(1)
            image_emb.append(temp_image_emb)

        image_emb = torch.cat(image_emb, dim = 1)
        img_pos_emb = self.position_embeddings(torch.tensor([pos for pos in range(len(image_emb[0]))]).repeat(len(image_emb), 1).to(GlobalConfig.device))
        turn_pos_emb = self.position_embeddings(torch.tensor([turn_id]*len(image_emb[0])).repeat(len(image_emb), 1).to(GlobalConfig.device))
        image_emb = img_pos_emb + turn_pos_emb + image_emb
        non_pad_mask = get_non_pad_mask(image_mask, self.padding_id)
        image_emb *= non_pad_mask
        image_emb = self.leaky_relu(image_emb)

        return sent_emb, image_emb

    def turn_level_self_attn(self, context_txt_emb, context_img_emb, turn_mask, image_mask, state_embedding):

        attn_mask = get_attn_key_pad_mask(seq_k=turn_mask, seq_q=turn_mask, padding_id=self.padding_id)
        non_pad_mask = get_non_pad_mask(turn_mask, self.padding_id)
        turn_len = len(context_txt_emb[0])
        pos_emb = self.position_embeddings(torch.tensor([pos for pos in range(turn_len)]).repeat(len(context_txt_emb), 1).to(GlobalConfig.device))
        context_txt_emb = context_txt_emb + pos_emb
        context_txt_emb, = self.context_text_self_attention(context_txt_emb, context_txt_emb, non_pad_mask, attn_mask)
        context_txt_emb = self.leaky_relu(context_txt_emb)

        attn_mask = get_attn_key_pad_mask(seq_k=image_mask, seq_q=turn_mask, padding_id=self.padding_id)
        non_pad_mask = get_non_pad_mask(turn_mask, self.padding_id)
        context_img_emb, = self.context_image_self_attention(context_img_emb, context_txt_emb, non_pad_mask, attn_mask)
        context_img_emb = self.leaky_relu(context_img_emb)

        # txt dst aware global truth
        context_txt_emb = self.get_state_aware(context_txt_emb[:, -1, :], state_embedding, 'txt')

        # img dst aware global truth
        context_img_emb = self.get_state_aware(context_img_emb[:, -1, :], state_embedding, 'img')

        return context_txt_emb, context_img_emb

    def get_item_embedding(self, attrs_id, image, state_embedding):
        # text emb
        text_embedding = self.get_text_emb(attrs_id, 'attrs')
        text_embedding = self.leaky_relu(torch.mean(text_embedding, dim = 1))

        # image emb
        image_embedding = self.get_image_emb(image)
        image_embedding = self.leaky_relu(image_embedding)

        text_embedding = self.get_state_aware(text_embedding, state_embedding, 'txt')

        image_embedding = self.get_state_aware(image_embedding, state_embedding, 'img')

        return text_embedding, image_embedding

    def get_kl_distribution(self, query_emb, key_emb):
        return self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(key_emb[0]), 1), key_emb, dim = -1))

    def cal_kl_loss(self, input, target):
        return torch.sum(self.kl(input, target), dim = -1)

    def get_kl_loss(self, target_txt_kl, target_img_kl, input_txt_kl, input_img_kl):

        text_klloss = self.cal_kl_loss(torch.log(target_txt_kl), input_txt_kl) + \
                      self.cal_kl_loss(torch.log(input_txt_kl), target_txt_kl)
        image_klloss = self.cal_kl_loss(torch.log(target_img_kl), input_img_kl) + \
                       self.cal_kl_loss(torch.log(input_img_kl), target_img_kl)

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

        # state embedding
        dst_ids = dst['input_ids'].to(GlobalConfig.device)
        dst_attention_mask = dst['attention_mask'].to(GlobalConfig.device)
        state_embedding = self.get_state_emb(dst_ids, dst_attention_mask, batch_size)

        # context emb
        context_text_embeddings = []
        context_image_embeddings = []

        for turn_id in range(DatasetConfig.dialog_context_size):
            context_text_embedding, context_image_embedding = self.get_context_emb(text_ids[:, turn_id, :], images[:, turn_id, :, :, :, :], image_mask[:, turn_id, :], turn_id)
            context_text_embeddings.append(context_text_embedding)
            context_image_embeddings.append(context_image_embedding)

        # dst or query aware
        image_mask = image_mask.view(batch_size, -1)
        context_txt_emb, context_img_emb = self.turn_level_self_attn(torch.cat(context_text_embeddings, dim = 1), torch.cat(context_image_embeddings, dim = 1), turn_mask, image_mask,
                                                   state_embedding)

        # kl learn loss
        hist_kl_learn_loss = 0
        prod_kl_learn_loss = []

        # context kl
        context_txt_emb, context_text_kl = context_txt_emb
        context_img_emb, context_image_kl = context_img_emb

        # sims
        neg_sims = []
        neg_kl_losses = []

        # pred_cl
        prod_txt_embeddings = []
        prod_img_embeddings = []
        # state attn
        neg_txt_attns = []
        neg_img_attns = []
        pos_txt_attns = []
        pos_img_attns = []

        # neg items
        for neg_id in range(DatasetConfig.max_neg_num):

            neg_text_embedding, neg_image_embedding = self.get_item_embedding(neg_attrs_id[:, neg_id, :],neg_imgs[:, neg_id, :, :, :],state_embedding)
            # neg sim
            neg_text_sim = cosine_similarity(context_txt_emb, neg_text_embedding[0], dim = -1)
            neg_image_sim = cosine_similarity(context_img_emb, neg_image_embedding[0], dim = -1)
            neg_sims.append(self.tanh(neg_text_sim + neg_image_sim))

            # # attn_graph
            # neg_txt_attns.append(neg_pred_text_embedding[1])
            # neg_img_attns.append(neg_pred_image_embedding[1])
            # neg_txt_attns.append(neg_text_embedding[1])
            # neg_img_attns.append(neg_image_embedding[1])

            if eval == False:
                # neg text image sim
                prod_txt_embeddings.append(neg_text_embedding[0])
                prod_img_embeddings.append(neg_image_embedding[0])
                # neg kl
                neg_text_kl = neg_text_embedding[1]
                neg_image_kl = neg_image_embedding[1]
                # neg kl loss
                neg_kl_loss = self.get_kl_loss(context_text_kl, context_image_kl, neg_text_kl, neg_image_kl)
                neg_kl_losses.append(neg_kl_loss)

        # loss component
        mask = get_mask(DatasetConfig.max_neg_num, neg_imgs_num, GlobalConfig.device)
        losses = [[] for _ in range(DatasetConfig.max_pos_num)]
        kl_losses = [[] for _ in range(DatasetConfig.max_pos_num)]

        # eval component
        rank_temp = torch.zeros(DatasetConfig.max_pos_num, batch_size, dtype=torch.long).to(GlobalConfig.device)

        for pos_id in range(DatasetConfig.max_pos_num):

            pos_text_embedding, pos_image_embedding = self.get_item_embedding(pos_attrs_id[:, pos_id, :],pos_imgs[:, pos_id, :, :, :],state_embedding)
            # pos sim
            pos_text_sim = cosine_similarity(context_txt_emb, pos_text_embedding[0], dim = -1)
            pos_image_sim = cosine_similarity(context_img_emb, pos_image_embedding[0], dim = -1)
            pos_sim = self.tanh(pos_text_sim + pos_image_sim)

            # # attn_graph
            # pos_txt_attns.append(pos_pred_text_embedding[1])
            # pos_img_attns.append(pos_pred_image_embedding[1])
            # pos_txt_attns.append(pos_text_embedding[1])
            # pos_img_attns.append(pos_image_embedding[1])

            if eval == False:
                # pos text image sim
                prod_txt_embeddings.append(pos_text_embedding[0])
                prod_img_embeddings.append(pos_image_embedding[0])
                # pos kl
                pos_text_kl = pos_text_embedding[1]
                pos_image_kl = pos_image_embedding[1]

                # pos kl loss
                pos_kl_loss = self.get_kl_loss(context_text_kl, context_image_kl, pos_text_kl, pos_image_kl)

            for neg_id in range(len(neg_sims)):
                # sim loss
                loss = torch.max(zeros, ones - pos_sim + neg_sims[neg_id])
                if eval:
                    rank_temp[pos_id] += torch.lt(pos_sim, neg_sims[neg_id]).long() * mask[:, neg_id]
                else:
                    loss = torch.max(zeros, ones - pos_sim + neg_sims[neg_id])

                    kl_loss = torch.max(zeros, ones - neg_kl_losses[neg_id] + pos_kl_loss)
                    kl_losses[pos_id].append(kl_loss)

                losses[pos_id].append(loss)

            # sim loss
            losses[pos_id] = torch.stack(losses[pos_id])
            losses[pos_id] = losses[pos_id].transpose(0, 1)
            losses[pos_id] = losses[pos_id].masked_select(mask.bool()).mean()

            if eval == False:
                # kl loss
                kl_losses[pos_id] = torch.stack(kl_losses[pos_id])
                kl_losses[pos_id] = kl_losses[pos_id].transpose(0, 1)
                kl_losses[pos_id] = kl_losses[pos_id].masked_select(mask.bool()).mean()

        loss = 0
        count = 0
        for loss_tmp in losses:
            if loss_tmp != []:
                count += 1
                loss += loss_tmp
        loss /= count

        sim_loss = float(loss)

        # # attn graph
        # neg_txt_attns = torch.stack(neg_txt_attns).transpose(0,1)
        # neg_img_attns = torch.stack(neg_img_attns).transpose(0,1)
        # pos_txt_attns = torch.stack(pos_txt_attns).transpose(0,1)
        # pos_img_attns = torch.stack(pos_img_attns).transpose(0,1)
        # context_text_kl = context_text_kl.unsqueeze(1)
        # context_image_kl = context_image_kl.unsqueeze(1)
        #
        # txt_attn_graph = torch.cat([context_text_kl, pos_txt_attns, neg_txt_attns], 1).cpu()
        # img_attn_graph = torch.cat([context_image_kl, pos_img_attns, neg_img_attns], 1).cpu()
        #
        # plt.figure()
        # ori_x_ticks = np.arange(0, 15, 1)
        # my_x_ticks = ['age', 'cat', 'br', 'sz', 'len', 'clr',
        #               'fit', 'sty', 'care', 'tp', 'mat', 'gdr', 'prt',
        #               'like', 'dislk']
        # ori_y_ticks = np.arange(0, 6, 1)
        # my_y_ticks = ['hist', 'pos', 'neg', 'neg', 'neg', 'neg']
        #
        # for batch_id in range(len(img_attn_graph)):
        #     plt.xticks(ori_x_ticks, my_x_ticks)
        #     plt.yticks(ori_y_ticks, my_y_ticks)
        #     plt.imshow((img_attn_graph[batch_id]+txt_attn_graph[batch_id]))
        #     plt.savefig('attn_graph/sum_attn_graph_{}.png'.format(batch_id), bbox_inches='tight')
        #     plt.clf()
        #
        # for batch_id in range(len(txt_attn_graph)):
        #     plt.xticks(ori_x_ticks, my_x_ticks)
        #     plt.yticks(ori_y_ticks, my_y_ticks)
        #     plt.imshow(txt_attn_graph[batch_id])
        #     plt.savefig('attn_graph/txt_attn_graph_{}.png'.format(batch_id), bbox_inches='tight')
        #     plt.clf()
        #
        # for batch_id in range(len(img_attn_graph)):
        #     plt.xticks(ori_x_ticks, my_x_ticks)
        #     plt.yticks(ori_y_ticks, my_y_ticks)
        #     plt.imshow(img_attn_graph[batch_id])
        #     plt.savefig('attn_graph/img_attn_graph_{}.png'.format(batch_id), bbox_inches='tight')
        #     plt.clf()

        # sims
        # overall_sims = torch.cat([pred_pos_sim.unsqueeze(1), torch.stack(pred_neg_sims, 0).transpose(0,1)], 1)

        if eval:
            return loss, rank_temp, pos_imgs_num
        else:
            # add text image sim loss
            if RecommendTrainConfig.txt_img_loss:
                prod_img_embeddings = torch.stack(prod_img_embeddings, 0).transpose(0,1)
                prod_txt_embeddings = torch.stack(prod_txt_embeddings, 0).transpose(0,1)
                prod_img_embeddings = nn.functional.normalize(prod_img_embeddings, p=2, dim=1)
                prod_txt_embeddings = nn.functional.normalize(prod_txt_embeddings, p=2, dim=1)
                pred_cl = torch.matmul(prod_txt_embeddings, prod_img_embeddings.transpose(1,2))
                labels = torch.tensor(range(len(pred_cl[0]))).repeat(batch_size, 1).to(GlobalConfig.device)
                cl_loss = self.ce(pred_cl, labels)
                pred_neg_mask = get_mask(DatasetConfig.max_neg_num, neg_imgs_num, GlobalConfig.device)
                pred_pos_mask = get_mask(DatasetConfig.max_pos_num, pos_imgs_num, GlobalConfig.device)
                pred_mask = torch.cat([pred_neg_mask, pred_pos_mask], dim=-1)
                txt_img_loss = cl_loss.masked_select(pred_mask.bool()).mean()
                loss += txt_img_loss * RecommendTrainConfig.text_image_sim_loss_weight

                # plt.figure()
                # for cl_id in range(len(pred_cl)):
                #     plt.imshow(self.softmax(pred_cl[cl_id]).cpu())
                #     plt.xticks([i for i in range(0, len(pred_cl[cl_id]), 1)])
                #     plt.yticks([i for i in range(len(pred_cl[cl_id]), 0, -1)])
                #     plt.savefig('attn_graph/txt_img_sim_{}.png'.format(cl_id), bbox_inches='tight')
                #     plt.clf()
            else:
                txt_img_loss = 0

            # add kl loss
            if RecommendTrainConfig.kl_diff_loss:
                kl_loss = 0
                count = 0
                for loss_tmp in kl_losses:
                    if loss_tmp != []:
                        count += 1
                        kl_loss += loss_tmp
                kl_loss /= count
                loss += kl_loss * RecommendTrainConfig.diff_loss_weight
            else:
                kl_loss = 0

            return loss, (sim_loss, txt_img_loss, kl_loss, 0, 0, 0)