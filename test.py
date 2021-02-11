import torch
from torch import nn
from meta import Meta
import numpy as np 
from MiniImagenet import MiniImagenet
from torch.utils.data import DataLoader
import random, sys, pickle
import argparse

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    print(args)
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    
    ckpt_dir = "./checkpoint_miniimage.pth"
    print("Load trained model")
    ckpt = torch.load(ckpt_dir)
    maml.load_state_dict(ckpt['model'])
    
    mini_test = MiniImagenet("F:\\ACV_project\\MAML-Pytorch\\miniimagenet\\", mode='test', n_way=args.n_way, k_shot=args.k_spt,
                k_query=args.k_qry,
                batchsz=1, resize=args.imgsz)

    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
    accs_all_test = []
    #count = 0
    #print("Test_loader",db_test)
    
    for x_spt, y_spt, x_qry, y_qry in db_test:
      
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
        x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
        accs_all_test.append(accs)

        # [b, update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Test acc:', accs)
        #count += 1

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    main()