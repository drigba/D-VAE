from __future__ import print_function
import os
from pickletools import optimize
import sys
import math
import pickle
import pdb
import networkx.utils as utils

import argparse
import random
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
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
from modelsRIGHT import *
from bayesian_optimization.evaluate_BN import Eval_BN
import networkx as nx
import json

parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='BN',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=7, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--max-n', type=int, default=7, help='number of vertices in the graphs')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait each time to save model states')

parser.add_argument('--sample-number', type=int, default=20, metavar='N',
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
parser.add_argument('--model', default='DVAE_NOBATCHNORM', help='model to use: DVAE, DVAE_NOBATCHNORM')

parser.add_argument('--hs', type=int, default=512, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=7, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--beta', type=int, default=0.01, metavar='S',
                    help='KL divergence weight in loss (default:0.01)')
parser.add_argument('--save-start', type=int, default=0, metavar='N',
                    help='how many epochs to wait to start saving model states')   
parser.add_argument('--early-stop-patience', type=int, default=50, metavar='S',
                    help='Patience before early stopping (default:10)')



# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size during training')




parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')   
parser.add_argument('--infer-batch-size', type=int, default=32, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


'''Prepare data'''

counter = 1
model_name = "NObatch" if args.model == "DVAE_NOBATCHNORM" else "YESbatch"
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'vertex_{}_{}dim_latent_KL{}_{}_hs{}_batchsize{}/{}{}_{}'.format(args.nvt-2,args.nz,args.beta,model_name,args.hs,args.batch_size,args.data_name, 
                                                                 args.save_appendix, counter))



while(os.path.exists(args.res_dir)):
    counter += 1
    args.res_dir = args.res_dir[:-1]
    args.res_dir = args.res_dir + str(counter)


args.scheduler_dir = os.path.join(args.res_dir, "scheduler")
args.optimizer_dir = os.path.join(args.res_dir, "optimizer")
args.model_dir = os.path.join(args.res_dir, "model")
args.latent_dir = os.path.join(args.res_dir, "latent")
args.fig_dir = os.path.join(args.res_dir, "figures")

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not os.path.exists(args.scheduler_dir):
    os.makedirs(args.scheduler_dir) 
if not os.path.exists(args.optimizer_dir):
    os.makedirs(args.optimizer_dir)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir) 
if not os.path.exists(args.latent_dir):
    os.makedirs(args.latent_dir) 
if not os.path.exists(args.fig_dir):
    os.makedirs(args.fig_dir) 
# delete old files in the result directory
remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and 
        not f.startswith('train_graph') and not f.startswith('test_graph') and
        not f.endswith('.pth')]
for f in remove_list:
    tmp = os.path.join(args.res_dir, f)
    if not os.path.isdir(tmp) and not args.keep_old:
        os.remove(tmp)

if not args.keep_old:
    # backup current .py files
    copy('train.py', args.res_dir)
    copy('modelsRIGHT.py', args.res_dir)
    copy('util.py', args.res_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    json.dump(args.__dict__, f, indent=2)
print('Command line input: ' + cmd_input + ' is saved.')




'''Prepare the model'''

# max_n = 5
# num_vertex_type = 1 or 5 + start and end so 3 or 7
# START_TYPE = 0
# END_TYPE = 1
# model = DVAE
# hs = 501 the default value  -hidden size of GRU
# nz = 56 the default latent space dimension
# bidirectional = False by default
# args.predictor = False
# args.continue_from = None
graph_args.max_n = args.nvt
graph_args.num_vertex_type = args.max_n
graph_args.START_TYPE = 0
graph_args.END_TYPE = 1



model = eval(args.model)(
        graph_args.max_n, 
        graph_args.num_vertex_type, 
        graph_args.START_TYPE, 
        graph_args.END_TYPE, 
        hs=args.hs, 
        nz=args.nz, 
        bidirectional=args.bidirectional,
        beta = args.beta
        )

# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50, verbose=True)
# scheduler = CyclicLR(optimizer, base_lr=args.lr, max_lr = 0.1, cycle_momentum=False)

model.to(device)


if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'))
else:
    if args.continue_from is not None:
        epoch = args.continue_from
        prev_dir = args.res_dir[:-1] + str(counter-1)
        load_module_state(model, os.path.join(prev_dir,"model", 
                                              'model_checkpoint{}.pth'.format(epoch)))
        load_module_state(optimizer, os.path.join(prev_dir,"optimizer", 
                                                  'optimizer_checkpoint{}.pth'.format(epoch)))
        load_module_state(scheduler, os.path.join(prev_dir, "scheduler",
                                                  'scheduler_checkpoint{}.pth'.format(epoch)))

# plot sample train/test graphs
def validate(epoch):
    model.eval()
    shuffle(valid_data)
    pbar = tqdm(valid_data)
    validation_loss = 0
    recon_loss = 0
    kld_loss = 0
    g_batch = []
    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            with torch.no_grad():
                g_batch = model._collate_fn(g_batch)
                mu, logvar = model.encode(g_batch)
                loss, recon, kld = model.loss(mu, logvar, g_batch)
                
                pbar.set_description('Validation, Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                    epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                    kld.item()/len(g_batch)))
                
                validation_loss += float(loss)
                recon_loss += float(recon)
                kld_loss += float(kld)
                g_batch = []

    print('====> Epoch: {} Average validation loss: {:.4f}'.format(
          epoch, validation_loss / len(valid_data)))
    return validation_loss, recon_loss, kld_loss

'''Define some train/test functions'''
def train(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            mu, logvar = model.encode(g_batch)
            loss, recon, kld = model.loss(mu, logvar, g_batch)
            
            pbar.set_description('Training, Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                kld.item()/len(g_batch)))
            loss.backward()
            
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            optimizer.step()
            g_batch = []

    
        
    print('====> Epoch: {} Average training loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))
    return train_loss, recon_loss, kld_loss



def visualize_recon(epoch):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    for i, g in enumerate(test_data[:1]):
        if args.model.startswith('SVAE'):
            g = g.to(device)
            g = model._collate_fn(g)
            g_recon = model.encode_decode(g)[0]
            g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)[0]
        elif args.model.startswith('DVAE'):
            g_recon = model.encode_decode(g)[0]
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)

        save0 =  os.path.join(args.fig_dir, name0)
        g_x = g.to_networkx()
        nx.draw_networkx(g_x)
        plt.show()
        plt.savefig(save0)
        plt.close()
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        save1 = os.path.join(args.fig_dir, name1)
        g_x_recon = g_recon.to_networkx()
        nx.draw_networkx(g_x_recon)
        plt.show()
        plt.savefig(save1)
        plt.close()



def test(fast_test = True):
    # test recon accuracy
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    n_perfect = 0
    print('Testing begins...')
    pbar = tqdm(test_data)
    g_batch = []
    for i, g in enumerate(pbar):
        # print("here")
        if args.model.startswith('SVAE'):
            g = g.to(device)
        g_batch.append(g)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g = model._collate_fn(g_batch)
            mu, logvar = model.encode(g)
            _, nll, _ = model.loss(mu, logvar, g)
            pbar.set_description('nll: {:.4f}'.format(nll.item()/len(g_batch)))
            Nll += nll.item()

            # construct igraph g from tensor g to check recon quality
            if fast_test:
                if args.model.startswith('SVAE'):  
                    g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)
                for _ in range(encode_times):
                    z = model.reparameterize(mu, logvar)
                    for _ in range(decode_times):
                        g_recon = model.decode(z)
                        n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))
            g_batch = []
    Nll /= len(test_data)
    if fast_test:
        acc = n_perfect / (len(test_data) * encode_times * decode_times)
        return Nll,acc
    else:
        print('Test average recon loss: {0}'.format(Nll))
        return Nll


def prior_validity(scale_to_train_range=False):
    if scale_to_train_range:
        Z_train, Y_train = extract_latent(train_data)
        z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    n_latent_points = 1000
    decode_times = 10
    n_valid = 0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    G_train = [g for g in train_data]
    if args.model.startswith('SVAE'):
        G_train = [g.to(device) for g in G_train]
        G_train = model._collate_fn(G_train)
        G_train = model.construct_igraph(G_train[:, :, :model.nvt], G_train[:, :, model.nvt:], False)
    pbar = tqdm(range(n_latent_points))
    cnt = 0
    for i in pbar:
        cnt += 1
        if cnt == args.infer_batch_size or i == n_latent_points - 1:
            z = torch.randn(cnt, model.nz).to(model.get_device())
            if scale_to_train_range:
                z = z * z_std + z_mean  # move to train's latent range
            for j in range(decode_times):
                g_batch = model.decode(z)
                G.extend(g_batch)
                for g in g_batch:
                    if is_valid_BN(g, graph_args.START_TYPE, graph_args.END_TYPE):
                        n_valid += 1
                        G_valid.append(g)

            cnt = 0
    r_valid = n_valid / (n_latent_points * decode_times)
    print('Ratio of valid decodings from the prior: {:.4f}'.format(r_valid))

    G_valid_str = [decode_igraph_to_ENAS(g) for g in G_valid]
    r_unique = len(set(G_valid_str)) / len(G_valid_str) if len(G_valid_str)!=0 else 0.0
    print('Ratio of unique decodings from the prior: {:.4f}'.format(r_unique))
    print(len(G_train))
    print(len(G_valid))
    r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))
    return r_valid, r_unique, r_novel


def extract_latent(data):
    model.eval()
    Z = []
    Y = []
    g_batch = []
    for i, g in enumerate(tqdm(data)):
        if args.model.startswith('SVAE'):
            g_ = g.to(device)
        elif args.model.startswith('DVAE'):
            # copy igraph
            # otherwise original igraphs will save the H states and consume more GPU memory
            g_ = g.copy()  
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
    return np.concatenate(Z, 0), np.array(Y)


'''Extract latent representations Z'''
def save_latent_representations(epoch):
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.latent_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.latent_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test
                         }
                     )
'''Training begins here'''
random.seed = 10
all_data = []
train_data = []
test_data = []
H_name = 'H_forward'  # name of the hidden states attribute

print("Loading train Data")
# During training vertices has to be processed in topological order
# Vertices are taken in order following their index
# Meaning that vertex with index 0 will be processed first
# Therefore the newly added starting node has to have index:0 otherwise an exception is thrown
i = 0
graph_data_path = os.path.join("..", "graph_data", "vertex_5")
for filename in tqdm(os.listdir(graph_data_path)):
    path = os.path.join(graph_data_path, filename)
    with open(path, 'rb') as pickle_file:
        # Load file
        graph = pickle.load(pickle_file)
        edge_list = graph.get_edgelist()
        # Create new graph
        graph2 = ig.Graph(directed=True)
        graph2.add_vertices(args.nvt)
        # Copy vertices to new graph
        for vs_i in range(len(graph.vs)):
            graph2.vs[vs_i+1]['type'] =  graph.vs[vs_i]['_nx_name']+2
        # Copy edges to new graph
        for edge_pair in edge_list:
            p1 = edge_pair[0]
            p2 = edge_pair[1]
            graph2.add_edge(p1+1,p2+1)
        # Set vertex attributes
        graph2.vs[0]['type'] = graph_args.START_TYPE
        graph2.vs[args.nvt-1]['type'] = graph_args.END_TYPE
        # graph2.add_edge(0,1)
        # graph2.add_edge(4,5)

        for vs_i,vs in enumerate(graph2.vs[1:-1]):
            if(len(vs.in_edges()) == 0):
                graph2.add_edge(0, vs_i+1)
            if(len(vs.out_edges()) == 0):
                graph2.add_edge(vs_i+1, len(graph2.vs)-1)
        all_data.append(graph2)


        


random.shuffle(all_data)
num_of_data = len(all_data)
num_of_train = math.floor(8*num_of_data/10)
remaining_num = num_of_data-num_of_train
num_of_test = math.floor(remaining_num/2)
train_data = all_data[:num_of_train]
valid_data = all_data[num_of_train:(num_of_train+num_of_test)]
test_data = all_data[(num_of_train+num_of_test):]
print(len(train_data))
print(len(valid_data))
print(len(test_data))




print("Train data loaded")

min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

if args.only_test:
    epoch = args.continue_from
    #sampled = model.generate_sample(args.sample_number)
    #save_latent_representations(epoch)
    #interpolation_exp2(epoch)
    #interpolation_exp3(epoch)
    #prior_validity(True)
    #test()
    #smoothness_exp(epoch, 0.1)
    #smoothness_exp(epoch, 0.05)
    #interpolation_exp(epoch)
    pdb.set_trace()

patience_counter = 0
early_stop_loss = math.inf
best_model = None
best_scheduler = None
best_optimizer = None
best_epoch = 0

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    train_loss, train_recon_loss, train_kld_loss = train(epoch)
    _, validation_recon, _ = validate(epoch)
    Nll = test(False)
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
            train_loss/len(train_data), 
            train_recon_loss/len(train_data), 
            train_kld_loss/len(train_data), 
            validation_recon/len(valid_data),
            Nll
            ))
    scheduler.step(train_loss)
    

    if epoch % args.save_interval == 0:
        print("plot train loss...")
        losses = np.loadtxt(loss_name)
        if losses.ndim == 1:
            continue
        fig = plt.figure()
        num_points = losses.shape[0]
        plt.plot(range(1, num_points+1), losses[:, 0], label='Train Total')
        plt.plot(range(1, num_points+1), losses[:, 1], label='Train Recon')
        plt.plot(range(1, num_points+1), losses[:, 2], label='Train KLD')
        plt.plot(range(1, num_points+1), losses[:, 3], label='Validation Recon')
        plt.plot(range(1, num_points+1), losses[:, 4], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.legend()
        plt.savefig(loss_plot_name)
        plt.close()
    # if epoch % (args.save_interval*5) == 0:
    #     Nll, acc = test()
    #     pred_rmse = 0
    #     r_valid, r_unique, r_novel = prior_validity(True)
    #     with open(test_results_name, 'a') as result_file:
    #         result_file.write("Epoch {} Test recon loss: {} recon acc: {:.4f} r_valid: {:.4f}".format(
    #                 epoch, Nll, acc, r_valid) + 
    #                 " r_unique: {:.4f} r_novel: {:.4f} pred_rmse: {:.4f}\n".format(
    #                     r_unique, r_novel, pred_rmse))

        

    if early_stop_loss > validation_recon:
        print("Validation loss decreased: {} -> {}".format(early_stop_loss,validation_recon))
        early_stop_loss = validation_recon
        patience_counter = 0
        # best_model = model.state_dict()
        # best_optimizer = optimizer.state_dict()
        # best_scheduler = scheduler.state_dict()
        best_epoch = epoch
        if epoch >=args.save_start:
            model_name = os.path.join(args.model_dir, 'model_checkpoint{}.pth'.format(epoch))
            optimizer_name = os.path.join(args.optimizer_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
            scheduler_name = os.path.join(args.scheduler_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_name)
            torch.save(optimizer.state_dict(), optimizer_name)
            torch.save(scheduler.state_dict(), scheduler_name)
            # print("visualize reconstruction examples...")
            visualize_recon(epoch)
            print("extract latent representations...")
            save_latent_representations(epoch)
            print("sample from prior...")
    else:
        patience_counter += 1
        print("Validation loss increased {} -> {}. Patience counter at: {}/{}".format(early_stop_loss,validation_recon, patience_counter, args.early_stop_patience))
    
    if patience_counter >= args.early_stop_patience:
        print("Early stopping at epoch: {}. Best results at epoch: {}".format(epoch, best_epoch))
        # model_name = os.path.join(args.res_dir, 'early_stopped_model_checkpoint{}.pth'.format(best_epoch))
        # optimizer_name = os.path.join(args.res_dir, 'early_stopped_optimizer_checkpoint{}.pth'.format(best_epoch))
        # scheduler_name = os.path.join(args.res_dir, 'early_stopped_scheduler_checkpoint{}.pth'.format(best_epoch))
        # torch.save(best_model, model_name)
        # torch.save(best_optimizer, optimizer_name)
        # torch.save(best_scheduler, scheduler_name)
        break
# interpolation_exp2(epoch)
# smoothness_exp(epoch)
# interpolation_exp3(epoch)

pdb.set_trace()
