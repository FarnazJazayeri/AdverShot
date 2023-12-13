import numpy as np


def basicblock(in_channel, out_channel, k_s_p=(3, 2, 0), main_layer='conv2d', activation='relu'):
    return [(main_layer, [out_channel, in_channel, k_s_p[0], k_s_p[0], k_s_p[1], k_s_p[2]]),  # c_out, c_in, k1, k2, s, p
            (activation, [True]),
            ('bn', [out_channel])]


def gateblock(hidden_channel, main_layer='conv2d', activation='sigmoid'):
    return [(main_layer, [hidden_channel, hidden_channel, 3, 3, 1, 1]),  # c_out, c_in, k1, k2, s, p
            ('bn', [hidden_channel]),
            (activation, [True])]


def create_config(num_layers, hidden_channel, out_channel, img_shape, gate_rnr):
    c, h, w = img_shape
    h_tmp = h
    max_layer = 0
    print(img_shape)
    config = []
    ### back_bone
    for l in range(num_layers):
        h_tmp /= 2
        if l == 0:
            c_in = c
            c_out = hidden_channel
        else:
            c_in = int(c_out)
            c_out = int(c_out / 2)
        if h_tmp >= 3:
            max_layer += 1
            block = basicblock(in_channel=c_in, out_channel=c_out, k_s_p=(3, 2, 1))
        else:
            block = basicblock(c_in, c_out, k_s_p=(3, 1, 1))
        config += block

    ### GRU
    if gate_rnr == "gru":
        block = basicblock(c_in, c_out * 2)
        config += block
        gate_z = gateblock(hidden_channel=c_out * 2, main_layer='conv2d_z', activation='sigmoid')
        gate_r = gateblock(hidden_channel=c_out * 2, main_layer='conv2d_r', activation='sigmoid')
        gate_h = gateblock(hidden_channel=c_out * 2, main_layer='conv2d_h', activation='sigmoid')
        config += gate_z
        config += gate_r
        config += gate_h
    ###
    ### head
    config.append(('flatten', []))
    h_out = int(np.ceil(h / 2 ** (max_layer)))
    print(h_out)
    c_in = c_out * h_out ** 2

    config.append(('linear', [out_channel, c_in]))
    return config


### Testing
# config = create_config(num_layers=3, hidden_channel=512, out_channel=5,
#                       img_shape=(3, 32, 32), gate_rnr="gru")
# print(config)

config_maml = [
    ### Backbone
    ('conv2d', [64, 1, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('conv2d', [64, 64, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('conv2d', [64, 64, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('conv2d', [64, 64, 2, 2, 1, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ### Head
    ('flatten', []),
    ('linear', [5, 64])  # args.n_way = 5
]

config_proto = [
    ('conv2d', [64, 1, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('conv2d', [64, 64, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('conv2d', [64, 64, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('conv2d', [64, 64, 2, 2, 1, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('flatten', []),
    ('linear', [64, 64])
]

config_maml_resnet_18 = [
    ('backbone_resnet_18', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_18', [64, 5])  # hidden_dim, n_way
]

config_maml_resnet_50 = [
    ('backbone_resnet_50', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_50', [64, 5])  # hidden_dim, n_way
]

config_proto_resnet_18 = [
    ('backbone_resnet_18', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_18', [64, 5])  # hidden_dim, n_way
]

config_proto_resnet_50 = [
    ('backbone_resnet_50', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_50', [64, 64])  # hidden_dim, emb_dim
]

config_proto_proposed = [
    ('backbone_resnet_18', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_18', [64, 64]),  # hidden_dim, emb_dim
    ('fsr', [64, 64])]