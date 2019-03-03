"""Different data loaders for subsets of data
"""
import os
import h5py
#import torch
from torch.utils.data import Dataset
import numpy as np
MAX_RECORDS = 10000 # take no more than given number of records in 
TEST_SIZE = 2000 # size of test set

    
class GeneExpressionBasicDataset(Dataset):
    """dataset with gene expression
    1st implementation:
    all data are taken from expression file
    """

    def __init__(self, expression_file, gene_file=None, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.expression_file = expression_file
        if os.path.exists(expression_file):
            self.expression_file_handle = h5py.File(expression_file, 'r')
        self.train = train
        self.gene_ids = None
        if gene_file is not None and os.path.exists(gene_file):
            self.gene_ids = []
            with open(gene_file) as f:
                for line in f:
                    self.gene_ids.append(int(line.strip()))
        #print(extract_genes(self.expression_file_handle, self.gene_ids))
        
        #self.gene_list = gene_list # if not genes are given, use specific gene list

    def __len__(self):
        if self.train:
            return MAX_RECORDS 
        return TEST_SIZE


    def __getitem__(self, idx):
        if self.train:
            expression = self.expression_file_handle['data']['expression'][
                idx, self.gene_ids]
        else:
            expression = self.expression_file_handle['data']['expression'][
                idx + MAX_RECORDS, self.gene_ids]
        expression = np.log(1.0*expression/expression.sum()+0.001)
        features = [1.0]
        return expression, features
        
        
        
def get_train_batch(train_ds, pos_size=10, neg_size=10, nbatches=10):
    """simple and ugly generator to extract data in batches
    """
    n = len(train_ds)
    for i in range(nbatches):
        pos_ids = np.random.choice(n, pos_size)
        neg_ids_e = np.random.choice(n, neg_size)
        neg_ids_f = np.random.choice(n, neg_size)
        expressions = []
        features = []
        labels = []
        for i in pos_ids:
            (e, f) = train_ds[i]
            expressions.append(e)
            features.append(f)
            labels.append(1.0)
        for i, j in zip(neg_ids_e, neg_ids_f):
            (e, _) = train_ds[i]
            (_, f) = train_ds[j]
            expressions.append(e)
            features.append(f)
            labels.append(0.0)
        expressions = np.stack(expressions)
        features = np.stack(features)
        labels = np.asarray(labels)
        yield expressions, features, labels
    