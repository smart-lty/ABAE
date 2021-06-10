import torch as t


class Config:

    K = 3
    m = 2
    W = 'all'

    conv_num = K - 1
    conv_channel = 64

    lr = 0.001
    drop_out = 0.05

    max_epoch = 200

    nheads = 8

    k_split = 10

    task = 'graph'
    dataset = 'cora'
    input_features = 500

    loss1_weight = 0.1
    loss2_weight = 0.9

    seed = 777

    load_model_path = './checkpoints'
    dataset_dir = './datasets/'
    save_processed_dir = './datasets/'

    use_gpu = True
    device = t.device('cuda') if use_gpu else t.device('cpu')


opt = Config()
