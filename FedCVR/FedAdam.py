import os
import copy
from operator import add

import time
import pickle
import numpy as np
from torch.onnx.symbolic_opset9 import dim
from tqdm import tqdm
from torch import nn
import torch
# from resnet import ResNet18, ResNet50, ResNet34
# from tensorboardX import SummaryWriter
# from convlarge import convLarge
from options import args_parser
from update import test_inference, LocalUpdate_Avg
from models import CNNFashion_Mnist, client_model, ResNet18, Net2
from utils import init_seed, plot_results, get_dataset2, update_global_weights_Adam

NONIID = []

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    # logger = SummaryWriter('./logs')
    args = args_parser()
    m_name = "FedAdam"

    print("iid:", args.iid, "Q:", args.local_ep, "N:", args.num_users, "m:", args.num_pars)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # use_cuda = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gl = 0.1
    run_num = 1

    train_loss_avg = []
    train_acc_avg = []
    test_loss_avg = []
    test_acc_avg = []
    # load dataset and user groups
    # p_ratio is ni / n, user_groups is a dict with key being user id and value being the idx of labeled data
    train_dataset, test_dataset, user_groups, p_ratio = get_dataset2(args)
    print(p_ratio)

    for tt in range(1, run_num + 1):
        # args.local_ep is the number of local epochs: Q
        init_seed(tt)
        global_model = None
        if args.dataset == "mnist":
            # global_model = CNNMnist(args=args)
            global_model = client_model(args.dataset)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar10':
            if args.model == "cnn":
                # global_model = client_model(args.dataset)
                global_model = Net2()
            elif args.model == "resnet":
                global_model = ResNet18()
                # global_model = client_model("Resnet18")
            else:
                raise ValueError("unsupported model!!!")
            # global_model = client_model(args.dataset)
        else:
            exit('Error: unrecognized dataset!!!')

        # global_model = global_model.cuda()
        global_model = global_model.to(device)

        # initialize the global model and two momentum
        global_weights = global_model.state_dict()
        print_every = 1

        # initialize the local model and control variable (all 0) for each user
        local_weights = []
        moment1 = copy.deepcopy(global_weights)
        moment2 = copy.deepcopy(global_weights)
        for key in global_weights.keys():
            moment1[key] = torch.zeros_like(global_weights[key])
            moment2[key] = torch.zeros_like(global_weights[key])
        for idx in range(args.num_users):
            local_weights.append(copy.deepcopy(global_weights))
        train_loss, train_accuracy, test_accuracy, test_loss = [], [], [], []

        test_acc, test_loss_ep = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        test_loss.append(test_loss_ep)

        for round_idx in tqdm(range(args.rounds)):
            local_losses = []
            print(f'\n | Global Training Round : {round_idx + 1} |\n')

            global_model.train()
            # sample user by uniform sampling without replacement
            idxs_users = np.random.choice(range(args.num_users), args.num_pars, replace=False)
            for idx in idxs_users:
                # print("idx:",idx,"len_idx_l:",len(user_groups_l[idx]),"len_idx_u:",len(user_groups_u[idx]))
                local_model = LocalUpdate_Avg(args=args, dataset=train_dataset, data_idxs=user_groups[idx],
                                              global_weights=global_weights)
                w, loss = local_model.update_weights(global_round=round_idx)
                with torch.no_grad():
                    local_weights[idx] = w

                local_losses.append(loss)
                # print("idx:",idx)
            # global_old = copy.deepcopy(global_weights)
            global_weights, moment1, moment2 = update_global_weights_Adam(global_weights=global_weights,
                                                                          local_weights=local_weights,
                                                                          idxs_users=idxs_users,
                                                                          num_users=args.num_users,
                                                                          p_ratio=p_ratio,
                                                                          moment1=moment1,
                                                                          moment2=moment2,
                                                                          step_size=gl)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            test_acc, test_loss_ep = test_inference(args, global_model, test_dataset)
            test_accuracy.append(test_acc)
            test_loss.append(test_loss_ep)
            # print global training loss after every 'i' rounds
            if (round_idx + 1) % print_every == 0:
                print(f' \nAvg Training Stats after {round_idx + 1} global rounds:')
                print(f'Training local Loss : {train_loss[-1]}')
                print('Test Accuracy: {:.2f}% \n'.format(100 * test_accuracy[-1]))
            # NONIID[tt].append( 100*test_accuracy[-1] )

        if tt == 1:
            train_loss_avg = train_loss
            train_acc_avg = train_accuracy
            test_loss_avg = test_loss
            test_acc_avg = test_accuracy
        else:
            train_loss_avg = [x + y for x, y in zip(train_loss_avg, train_loss)]
            train_acc_avg = [x + y for x, y in zip(train_acc_avg, train_accuracy)]
            test_loss_avg = [x + y for x, y in zip(test_loss_avg, test_loss)]
            test_acc_avg = [x + y for x, y in zip(test_acc_avg, test_accuracy)]

    train_loss_avg = [tmp / run_num for tmp in train_loss_avg]
    train_acc_avg = [tmp / run_num for tmp in train_acc_avg]
    test_loss_avg = [tmp / run_num for tmp in test_loss_avg]
    test_acc_avg = [tmp / run_num for tmp in test_acc_avg]

    exp_name = args.exp_addName+args.dataset + "/" + m_name + "/" + str(args.model) \
               + ",#" + str(args.num_users) + ",lr=" + str(args.lr) \
               + ",global_lr=" + str(gl) + ",#epochs" + str(args.local_ep) \
               + ",num_par" + str(args.num_pars) + ", varepoch" + str(args.local_ep_var)

    plot_results(exp_name=exp_name, train_loss=train_loss_avg, train_accuracy=train_acc_avg,
                 test_loss=test_loss_avg, test_accuracy=test_acc_avg)
