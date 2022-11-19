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
import igraph as ig
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
parser.add_argument('--hs', type=int, default=150, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=16, metavar='N',
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










graph_args.max_n = 6
graph_args.num_vertex_type = 3
graph_args.START_TYPE = 0
graph_args.END_TYPE = 1

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

args.res_dir = os.path.join(args.file_dir, 'vertex_4_same_label_batch_2_data_35_smaller_model/{}{}'.format(args.data_name, 
                                                                 args.save_appendix))

all_data = []
i = 0
for filename in os.listdir("..\\graph_data\\vertex_4"):
    path = os.path.join("..\\graph_data\\vertex_4", filename)
    with open(path, 'rb') as pickle_file:
        # Load file
        graph = pickle.load(pickle_file)
        edge_list = graph.get_edgelist()
        # Create new graph
        graph2 = ig.Graph(directed=True)
        graph2.add_vertices(6)
        # Copy vertices to new graph
        for vs_i in range(len(graph.vs)):
            graph2.vs[vs_i+1]['type'] =  vs_i+2
        # Copy edges to new graph
        for edge_pair in edge_list:
            p1 = edge_pair[0]
            p2 = edge_pair[1]
            graph2.add_edge(p1+1,p2+1)
        # Set vertex attributes
        graph2.vs[0]['type'] = graph_args.START_TYPE
        graph2.vs[5]['type'] = graph_args.END_TYPE
        # graph2.add_edge(0,1)
        # graph2.add_edge(4,5)

        for vs_i,vs in enumerate(graph2.vs[1:-1]):
            if(len(vs.in_edges()) == 0):
                graph2.add_edge(0, vs_i+1)
            if(len(vs.out_edges()) == 0):
                graph2.add_edge(vs_i+1, len(graph2.vs)-1)
        all_data.append(graph2)






def visualize_recon(g):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    g_recon = model.encode_decode(g)[0]
    name0 = 'graph_epoch_original'
    plot_DAG(g, args.res_dir, name0, data_type=args.data_type)
    name1 = 'graph_epoch_recon'
    plot_DAG(g_recon, args.res_dir, name1, data_type=args.data_type)


    



min_var = 5000
min_var_ix = -1

for q in range(50,60,10): 
    invalids = []                                                                       
    p = os.path.join(args.res_dir, 'model_checkpoint'+str(q)+'.pth')
    load_module_state(model, p)
    occurences = []
    # graph_list = []
    ic = 0
    invalid = 0
    print("START")
    for ix, graph_occ in enumerate(all_data):
        graph = graph_occ[0]
        graph = [graph]
        graph = model._collate_fn(graph)
        g_recons = []
        for decodenumber in range(1):
            g_recons.append(model.encode_decode(graph)[0])
        
        g_recon = g_recons[0]

        # print(graph[0])
        # for vs in graph[0].vs:
        #     print(vs["type"])
        # print(g_recon)
        # for vs in g_recon.vs:
        #     print(vs["type"])
        # print(util.is_same_DAG(graph[0],g_recon))
        # print("----")
        # input()
        ix_list = []
        for vs in g_recon.vs:
            if vs["type"] == graph_args.END_TYPE or vs["type"] == graph_args.START_TYPE:
                ix_list.append(vs.index)
        g_s = g_recon.copy()
        g_recon.delete_vertices(ix_list)

        b = False

        for ix2, g2 in enumerate(all_data):
            ix_list = []

            for vs in g2[0].vs:
                 if vs["type"] == graph_args.END_TYPE or vs["type"] == graph_args.START_TYPE:
                    ix_list.append(vs.index)
            g2[0].delete_vertices(ix_list)
            if util.is_same_DAG(g2[0],g_recon):
                all_data[ix2][1] += 1
                print("Indexes: " + str(ix) + " - " +str(ix2))
                b = True
                # break
        if not b:
            invalids.append(graph[0])
            invalid += 1

        

        

    not_null = 0
    max = 0
    max_g = ""
    for gm in all_data:
        # print(gm[1])
        if gm[1] > 0:
            not_null +=1

        if gm[1] > max:
            max = gm[1]
            max_g = gm[0]
    print("---------")
    occ = [gm[1] for gm in all_data]
    occ_var = statistics.variance(occ)
    if occ_var < min_var:
        min_var = occ_var
        min_var_ix = q
    print(occ)
    # for g in invalids:
    #     g_re = model.encode_decode(g)
    #     print(model.encode(g_re))
    #     print(g)
    #     print(g_re[0])
        # input()
    print(not_null)
    print(max)
    print(max_g)
    print(invalid)

    for gm in all_data:
        gm[1] = 0

    
print(min_var)
print(min_var_ix)

