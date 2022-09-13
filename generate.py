from __future__ import print_function
import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models import *
from bayesian_optimization.evaluate_BN import Eval_BN
from igraph import *
import networkx as nx
from collections import Counter
import statistics
import util













parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='ENAS',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=10000, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='DVAE', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')
parser.add_argument('--predictor', action='store_true', default=False,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=5, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)

graph_args.max_n = 4
graph_args.num_vertex_type = 6
graph_args.START_TYPE = 0
graph_args.END_TYPE = 5

model = eval(args.model)(
        graph_args.max_n, 
        graph_args.num_vertex_type, 
        graph_args.START_TYPE, 
        graph_args.END_TYPE, 
        hs=args.hs, 
        nz=args.nz, 
        bidirectional=args.bidirectional
        )



model.to(device)
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

args.res_dir = os.path.join(args.file_dir, 'vertex_4_smaller_lr/{}{}'.format(args.data_name, 
                                                                 args.save_appendix))

all_data = []

for filename in os.listdir("..\\graph_data\\vertex_4"):
    path = os.path.join("..\\graph_data\\vertex_4", filename)
    with open(path, 'rb') as pickle_file:
        graph = pickle.load(pickle_file)
        for v in graph.vs:
            v['type'] = v['_nx_name']
        all_data.append([graph,0])


for q in range(50,110,5):                                                                        
    p = os.path.join(args.res_dir, 'model_checkpoint'+str(q)+'.pth')
    load_module_state(model, p)
    occurences = []
    # graph_list = []

    print("START")

    for i in range(4):
        sampled = model.generate_sample(args.sample_number)
        for _, gr_l in enumerate(tqdm(sampled)) :
            cont = False
            for v in gr_l.vs:
                v['_nx_name'] = v['type']
            for gr in all_data:
                # if len(gr[0].edges) == len(gr_lx.edges):
                #     print(gr[0].nodes.data(True))
                if is_same_DAG(gr[0],gr_l):
                    gr[1] += 1
                    cont = True
                    break
            input()
            # if not cont:
            #     graph_list.append([gr_lx,1])


    max = 0
    max_g = 0
    li = []
    for qwe in all_data:
        occurences.append(qwe[1])
        li.append(qwe[1])
        if qwe[1] > max:
            max = qwe[1]
            max_g = qwe[0]
    print("****")
    # nx.draw_networkx(max_g)

    plt.plot(occurences)
    plt.savefig("vertex_4_smaller_lr/generation/occurences_"+str(q))
    plt.close()

    plt.hist(occurences, bins=20)
    plt.savefig("vertex_4_smaller_lr/generation/frequency_"+str(q))
    plt.close()


    print(len(occurences))
    print(max)
    print(statistics.variance(li))
    print(mean(li))
    print(median(li))

