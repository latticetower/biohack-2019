import os
import sys
#import time
#import random
import argparse
import itertools
import dataloader
import torch
from torch.autograd import Variable
import torch.utils.data
from model import BasicModel
import torch.nn as nn
import model
from dataloader import get_train_batch

learning_rate = 0.0002
betas = (0.5, 0.999)

def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.to(torch.device('cuda:0'))
    return Variable(x)
    
    
#assert os.getenv('CUDA_VISIBLE_DEVICES')

parser = argparse.ArgumentParser()
find_action = type('', (argparse.Action, ), 
    dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset', 
        choices=dict(
           basic_dataset=dataloader.GeneExpressionBasicDataset,
        #cub2011 = cub2011.CUB2011MetricLearning, 
        #cars196 = cars196.Cars196MetricLearning, 
        #stanford_online_products = stanford_online_products.StanfordOnlineProducts
        ), 
        default = dataloader.GeneExpressionBasicDataset, 
    action=find_action)
parser.add_argument("--expression_file",
    default="../../biohack-2019/mouse_matrix.h5"
)
parser.add_argument("--gene_file",
    default="../data/kinase_indices.txt"
)

parser.add_argument('--data', default = 'data')
parser.add_argument('--savepath', default='saves')
parser.add_argument('--log', default = 'data/log.txt')
parser.add_argument('--seed', default = 1, type = int)
parser.add_argument('--threads', default = 16, type = int)
parser.add_argument('--epochs', default = 100, type = int)
parser.add_argument('--batch', default = 128, type = int)
parser.add_argument('--nbatches', default = 10, type = int)

opts = parser.parse_args()
if not os.path.exists(opts.savepath):
    os.makedirs(opts.savepath)
    
print(os.path.exists(opts.expression_file))
train_ds = opts.dataset(opts.expression_file, gene_file=opts.gene_file, train=True)
val_ds = opts.dataset(opts.expression_file, gene_file=opts.gene_file, train=False)


#print(next(get_train_batch(train_ds)))
model = BasicModel(len(train_ds.gene_ids), 1)
#model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)


for epoch in range(opts.epochs):
    print("Epoch %s" % epoch)
    for i, (expressions, features, labels) in enumerate(
            get_train_batch(train_ds, pos_size=opts.batch//2, 
                neg_size=opts.batch//2, nbatches=10)):
        optimizer.zero_grad()

        e = to_var(torch.from_numpy(expressions).float())
        f = to_var(torch.from_numpy(features).float())
        l = to_var(torch.from_numpy(labels).float())
        out = model.forward(e, f)
        loss = nn.MSELoss()(out, l)
        loss.backward()
        print(to_np(loss))
        optimizer.step()
    
    path = os.path.join(opts.savepath, "emb_epoch_%s.model" % (epoch))
    
    torch.save(model.state_dict(), path)