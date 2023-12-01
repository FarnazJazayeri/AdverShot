


config_maml = [
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
            ('linear', [5, 64]) # args.n_way = 5
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
    ('head_resnet_18', [64, 5]) # hidden_dim, n_way
]

config_maml_resnet_50 = [
    ('backbone_resnet_50', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_50', [64, 5]) # hidden_dim, n_way
]

config_proto_resnet_18 = [
    ('backbone_resnet_18', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_18', [64, 5]) # hidden_dim, n_way
]

config_proto_resnet_50 = [
    ('backbone_resnet_50', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_50', [64, 64]) # hidden_dim, emb_dim
]

config_proto_proposed = [
    ('backbone_resnet_18', [1, 64]),  # in_dim, hidden_dim
    ('head_resnet_18', [64, 64]),  # hidden_dim, emb_dim
    ('fsr', [64, 64])]