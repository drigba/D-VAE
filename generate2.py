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
from models2 import *
from bayesian_optimization.evaluate_BN import Eval_BN
from igraph import *
import networkx as nx
from collections import Counter
import statistics
import util


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='BN',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=True,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='DVAE_BN', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=2000, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=256, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=64, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')
parser.add_argument('--predictor', action='store_true', default=False,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=6000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
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
graph_args.num_vertex_type = 6
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

args.res_dir = os.path.join(args.file_dir, 'vertex_4/{}{}'.format(args.data_name, 
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


def interpolation_great_circle(ix,g):
    interpolate_number = 36
    z0, _ = model.encode(g)
    norm0 = torch.norm(z0)
    z1 = torch.ones_like(z0)
    dim_to_change = random.randint(0, z0.shape[1]-1)  # this to get different great circles
    print(dim_to_change)
    z1[0, dim_to_change] = -(z0[0, :].sum() - z0[0, dim_to_change]) / z0[0, dim_to_change]
    z1 = z1 / torch.norm(z1) * norm0
    print('z0: ', z0, 'z1: ', z1, 'dot product: ', (z0 * z1).sum().item())
    print('norm of z0: {}, norm of z1: {}'.format(norm0, torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    omega = torch.acos(torch.dot(z0.flatten(), z1.flatten()))
    print('angle between z0 and z1: {}'.format(omega))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        theta = 2*math.pi / interpolate_number * j
        zj = z0 * np.cos(theta) + z1 * np.sin(theta)
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)
    for ix_i,g_s in enumerate(G):
        name0 = 'graph_id{}_sampled{}'.format(ix,ix_i)
        save0 =  os.path.join(args.res_dir,"interpolated", str(ix))
        if not os.path.exists(save0):
            os.makedirs(save0)
        final_save = os.path.join(save0,name0)
        my_plot_DAG(final_save, g_s)

def my_plot_DAG(save_path, g):
    g_x = g.to_networkx()
    nx.draw_networkx(g_x)
    plt.show()
    plt.savefig(save_path)
    plt.close()

def visualize_recon(ix,g):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    g_recon = model.encode_decode(g)[0]

    name0 = 'graph_id{}_original'.format(ix)
    save0 =  os.path.join(args.res_dir,"graphs", name0)
    print(save0)
    # plot_DAG(g, args.res_dir, name0, data_type=args.data_type)
    g_x = g.to_networkx()
    nx.draw_networkx(g_x)
    plt.show()
    plt.savefig(save0)
    plt.close()
    name1 = 'graph_id{}_recon'.format(ix)
    save1 = os.path.join(args.res_dir,"graphs", name1)
    g_x_recon = g_recon.to_networkx()
    nx.draw_networkx(g_x_recon)
    plt.show()
    plt.savefig(save1)
    plt.close()


q = 6000
p = os.path.join(args.res_dir, 'model_checkpoint'+str(q)+'.pth')
load_module_state(model,p)

# for ix,g in enumerate(all_data):
#     print("Visualizing graph_{}: ".format(ix) + str(g))
#     visualize_recon(ix,g)
#     input()

G = model.generate_sample(10)

for ix,g in enumerate(G):
    print("Printing graph:{}\n{}".format(ix,g))
    interpolation_great_circle(ix,g)
    # name0 = 'graph_id{}_sampled'.format(ix)
    # save0 = save0 =  os.path.join(args.res_dir,"sampled", name0)
    # g_x = g.to_networkx()
    # nx.draw_networkx(g_x)
    # plt.show()
    # plt.savefig(save0)
    # plt.close()
    input()
    

        

        

