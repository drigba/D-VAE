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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from modelsRIGHT import *
from igraph import *
import networkx as nx
from collections import Counter
import statistics
import util
import itertools
from sklearn.decomposition import PCA
import dotmap
from dotmap import DotMap
import pandas as pd
import json
from scipy.stats import wasserstein_distance
from wrapper import GraphWrapper




def prior_validity(model, latents, data, batch_size, latent_points):
    Z_train = latents
    Zs_valid = []
    Zs_invalid = []
    Zs_plus_decode = []
    Wrapped_G = []
    z_mean, z_std = np.array(Z_train.mean(0)) ,np.array(Z_train.std(0)) 
    # print(z_mean)
    # print(z_std)
    z_mean, z_std = torch.FloatTensor(z_mean).to(model.get_device()), torch.FloatTensor(z_std).to(model.get_device())
    n_latent_points = latent_points
    decode_times = 10
    n_valid = 0
    print('Prior validity experiment begins...')
    
    G_valid = []
    G_train = [g for g in data]
    pbar = tqdm(range(n_latent_points))
    cnt = 0
    G_valid = []
    G_invalid = []
    n_valid = 0
    for i in pbar:
        cnt += 1
        if cnt == batch_size or i == n_latent_points - 1:
            z = torch.randn(cnt, model.nz).to(model.get_device())
        # print(z)
            z = z * z_std + z_mean  # move to train's latent range
        # print(z)
            
            # print(z.cpu().detach().numpy())

            for j in range(decode_times):
                g_batch = model.decode(z)
                for ix,g in enumerate(g_batch):
                    if is_valid_BN(g, model.START_TYPE, model.END_TYPE,model.nvt):
                        n_valid +=1
                        G_valid.append(g)
                        Wrapped_G.append(GraphWrapper(g))
                        Zs_valid.append(z.cpu().detach().numpy()[ix])
                    else:
                        G_invalid.append(g)
                        Zs_invalid.append(z.cpu().detach().numpy()[ix])


            cnt = 0
    return G_valid, G_invalid, Zs_valid, Zs_invalid, Wrapped_G


def interpolate1_randomPoints( model, latents,n_split):
    z1 = torch.randn(1,model.nz).to(model.get_device())
    z2 = torch.randn(1,model.nz).to(model.get_device())
    Z_train = latents
    z_mean, z_std = np.array(Z_train.mean(0)) ,np.array(Z_train.std(0)) 
    z_mean, z_std = torch.FloatTensor(z_mean).to(model.get_device()), torch.FloatTensor(z_std).to(model.get_device())
    z1 = z1 * z_std + z_mean
    z2 = z2 * z_std + z_mean
    z_diff = (z1-z2)/n_split
    points = []
    for i in range(n_split+1):
        offset = z_diff*i
        c_z = offset+z2
        print(c_z.size())
        points.append(c_z.cpu().detach().numpy())
        g = model.decode(c_z)
        g_x = g[0].to_networkx()
        pos = nx.circular_layout(g_x)
        nx.draw_networkx(g_x, pos=pos)
        plt.show()
    points = np.array(points)
    print(np.shape(points))
    points = points.reshape((n_split+1,model.nz))
    pca = PCA(n_components=2)
    components = pca.fit_transform(points)
    plt.scatter(components[:,0], components[:,1])
    plt.scatter(points[:,0], points[:,1])
    for point in points:
        print(point)
    for component in components:
        print(component)
    return components


def interpolate1_RandomPoints_B( model, latents,n_split,decode_times):
    z1 = torch.randn(1,model.nz).to(model.get_device())
    z2 = torch.randn(1,model.nz).to(model.get_device())
    Z_train = latents
    z_mean, z_std = np.array(Z_train.mean(0)) ,np.array(Z_train.std(0)) 
    z_mean, z_std = torch.FloatTensor(z_mean).to(model.get_device()), torch.FloatTensor(z_std).to(model.get_device())
    z1 = z1 * z_std + z_mean
    z2 = z2 * z_std + z_mean
    z_diff = (z1-z2)/n_split
    points = []
    for i in range(n_split+1):
        offset = z_diff*i
        c_z = offset+z2
        print(c_z.size())
        points.append(c_z.cpu().detach().numpy())
        # g = model.decode(c_z)
        g, _ = decode_from_latent_space(c_z,model,decode_times,'variable',True, 'BN')
        g_x = g[0].to_networkx()
        pos = nx.circular_layout(g_x)
        nx.draw_networkx(g_x, pos=pos)
        plt.show()
    points = np.array(points)
    print(np.shape(points))
    points = points.reshape((n_split+1,model.nz))
    pca = PCA(n_components=2)
    components = pca.fit_transform(points)
    plt.scatter(components[:,0], components[:,1])
    plt.scatter(points[:,0], points[:,1])
    for point in points:
        print(point)
    for component in components:
        print(component)
    return components

def interpolate2_randomGraphs(model, data,latents,n_split, decode_times):
    g_s = random.sample(data,2)
    z1 = model.encode(g_s[0])[0]
    z2 = model.encode(g_s[1])[0]
    Z_train = latents
    z_mean, z_std = np.array(Z_train.mean(0)) ,np.array(Z_train.std(0)) 
    z_mean, z_std = torch.FloatTensor(z_mean).to(model.get_device()), torch.FloatTensor(z_std).to(model.get_device())
    z1 = z1 * z_std + z_mean
    z2 = z2 * z_std + z_mean
    z_diff = (z1-z2)/n_split
    points = []
    for i in range(n_split+1):
        offset = z_diff*i
        c_z = offset+z2
        print(c_z.size())
        points.append(c_z.cpu().detach().numpy())
        # g = model.decode(c_z)
        g, _ = decode_from_latent_space(c_z,model,decode_times,'variable',True, 'BN')
        g_x = g[0].to_networkx()
        pos = nx.circular_layout(g_x)
        nx.draw_networkx(g_x, pos=pos)
        plt.show()
    points = np.array(points)
    print(np.shape(points))
    points = points.reshape((n_split+1,model.nz))
    pca = PCA(n_components=2)
    components = pca.fit_transform(points)
    plt.scatter(components[:,0], components[:,1])
    plt.scatter(points[:,0], points[:,1])
    for point in points:
        print(point)
    for component in components:
        print(component)
    return components


def interpolate3_Circle(model,data, interpolate_number,decode_times):
    print('Interpolation experiments around a great circle')
    # interpolation_res_dir = res_dir
    # if not os.path.exists(interpolation_res_dir):
    #     os.makedirs(interpolation_res_dir) 
    model.eval()
    g_ix = random.randint(0,len(data)-1)
    print(len(data))
    print(g_ix)
    g0 =  data[g_ix]
    z0, _ = model.encode(g0)
    norm0 = torch.norm(z0)
    z1 = torch.ones_like(z0)
    # there are infinite possible directions that are orthogonal to z0,
    # we just randomly pick one from a finite set
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
    G, _ = decode_from_latent_space(latent_points=Z,model= model,decode_attempts=decode_times, return_igraph=True, data_type='BN') 
    for j in range(0, interpolate_number + 1):
        g_x = G[j].to_networkx()
        pos = nx.circular_layout(g_x)
        nx.draw_networkx(g_x, pos=pos)
        plt.show()



def is_same_DAG2(g0, g1):
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    if set(g0.vs["type"]) != set(g1.vs["type"]):
        return False
    if g0.ecount() != 0 or g1.ecount() == 0:
        for vi in range(g0.vcount()):
            g0_index_of_type = g0.vs.find(type=vi+2)
            g1_index_of_type = g1.vs.find(type=vi+2)
            # g0_vs = g0.vs[g0_index_of_type]
            # g1_vs = g1.vs[g1_index_of_type]

            # if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            #     return False
            g0_neighbours = set([g0.vs[vs_i]["type"] for vs_i in g0.neighbors(g0_index_of_type, 'in')])
            g1_neighbours = set([g1.vs[vs_i]["type"] for vs_i in g1.neighbors(g1_index_of_type, 'in')])
            if g0_neighbours != g1_neighbours:
                return False
    return True

def visualize_recon(model,data,epoch, fig_dir):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    for i, g in enumerate(data[:1]):
        g_recon = model.encode_decode(g)[0]
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)

        save0 =  os.path.join(fig_dir, name0)
        g_x = g.to_networkx()
        nx.draw_networkx(g_x)
        plt.show()
        plt.savefig(save0)
        plt.close()
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        save1 = os.path.join(fig_dir, name1)
        g_x_recon = g_recon.to_networkx()
        nx.draw_networkx(g_x_recon)
        plt.show()
        plt.savefig(save1)
        plt.close()

def DAG_hash(graph) -> int:
        g = graph
        nodeTypes = sorted(g.vs["type"])
        n2 =  "".join([str(nodeType) for nodeType in nodeTypes + [0]] + [str(neighbour)  for nodeType in nodeTypes for neighbour in sorted([g.vs[nodeIndex]["type"] for nodeIndex in g.neighbors(g.vs.find(type = nodeType), 'in')])+[0]])
        return int(n2)