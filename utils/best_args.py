best_args = {
    'gearbox-sdust': {

        'p2pchf': {
            -1: {
                'communication_epoch': 30,
                'local_lr': 0.0001,  # 客户端学习率
                'public_lr': 0.0001,  # 公共端学习率
                'local_epoch': 15,  # 客户端训练集训练轮数
                'public_epoch': 2,  # 共享训练集训练轮数
                'public_batch_size': 64,
                'local_batch_size': 64,
                'temp': 0.2,  # 本地训练蒸馏损失
                'fccm': 0.0051,
                'fisl': 1.5# 实例损失函数权重
            },
        },

    },
    'gearbox-bjut': {
        'p2pchf': {
            -1: {
                'communication_epoch': 30,
                'local_lr': 0.0001,  # 客户端学习率
                'public_lr': 0.0001,  # 公共端学习率
                'local_epoch': 15,  # 客户端训练集训练轮数
                'public_epoch': 2,  # 共享训练集训练轮数
                'public_batch_size': 64,
                'local_batch_size': 64,
                'temp': 0.2,  # 本地训练蒸馏损失
                'fccm': 0.0051,
                'fisl': 1.5  # 实例损失函数权重
            },
        },
    },
    'bearing-hust': {

        'p2pchf': {
            -1: {
                'communication_epoch': 30,
                'local_lr': 0.0001,  # 客户端学习率
                'public_lr': 0.0001,  # 公共端学习率
                'local_epoch': 15,  # 客户端训练集训练轮数
                'public_epoch': 2,  # 共享训练集训练轮数
                'public_batch_size': 64,
                'local_batch_size': 64,
                'temp': 0.2,  # 本地训练蒸馏损失
                'fccm': 0.0051,
                'fisl': 1.5# 实例损失函数权重
            },
        },

    },






}
