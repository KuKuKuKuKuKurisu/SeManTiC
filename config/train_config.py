from utils.better_abc import ABCMeta, abstract_attribute

class TrainConfig(metaclass=ABCMeta):
    batch_size = abstract_attribute()
    num_iterations = abstract_attribute()
    learning_rate = abstract_attribute()
    print_freq = abstract_attribute()
    valid_freq = abstract_attribute()
    patience = abstract_attribute()
    num_data_loader_workers = abstract_attribute()

class RecommendTrainConfig(TrainConfig):

    # simmc
    simmc2 = False
    # using learn
    using_learn = True

    # few shot
    few_shot = False
    fewshot_epochs = 1
    dst_data_proportion = 0.01

    # full data
    full_data = False

    batch_size = 64
    num_iterations = 50
    dropout = 0.1
    soft_max_temperture = 0.5

    if simmc2:
        learning_rate = 4e-5 #85
        print_freq = 64
        valid_freq = 128
    else:
        learning_rate = 5e-4
        print_freq = 128
        valid_freq = 256
    patience = 50
    fine_tune_patience = 5
    num_data_loader_workers = 4
    gradient_clip = False
    max_gradient_norm = 1
    image_encoder = 'resnet18'

    # support losses
    # text image sim loss
    txt_img_loss = True
    txt_img_threshold = 1.0
    text_image_sim_loss_weight = 1

    # kl loss
    kl_diff_loss = True
    diff_loss_weight = 1

    # learn loss
    start_learn = False
    start_learn_epoch = 1
    learn_loss_weight = 0.5

    # DST
    DST = True

    # delete
    attribute_aware = False
    using_threshold = True
    pretrain = False