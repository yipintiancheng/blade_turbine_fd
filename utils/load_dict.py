# 模型载入策略，一个比较稳健的策略，载入基本不会报错，并且可以特制化是否需要某些层


def load(net, ckpt_path, ignore_list=None, display=True):
    if ignore_list is None:
        ignore_list = []
    #network_params = self.config['network']
    ckpt = torch.load(ckpt_path)# 载入权重, map_location=torch.device(self.config['network']['device'])
    state_dict = ckpt['state_dict']
    if network_params['use_parallel']:# 并行
        model_dict = net.module.state_dict()
    else:
        model_dict = net.state_dict()
    for key in list(state_dict.keys()):
        res = True
        for rule in ignore_list:# 确定这一层要不要去掉，写成循环的形式可以不用担心输入顺序
            if key.startswith(rule):# Python startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
                res = False
                break
        if res:# 如果没有被ignore
            if key in model_dict.keys():# 还要判断模型里有没有这一层
                if display:# 这个决定是否要print
                    print("Loading parameter {}".format(key))
                model_dict[key] = state_dict[key]
    if network_params['use_parallel']:# 并行
        net.module.load_state_dict(model_dict)
    else:
        net.load_state_dict(model_dict)
    print(">>> Loading model successfully from {}.".format(ckpt_path))