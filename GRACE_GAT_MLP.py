import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator, from_predefined_split, from_predefined_split_wiki, \
    from_predefined_split_heter, from_predefined_split_heterophilous
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv, GATConv, GINConv, Node2Vec, SAGEConv
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, Actor
from torch_geometric.datasets import WebKB
from ogb.nodeproppred import NodePropPredDataset
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import time
from dataset import WikipediaNetwork, load_fixed_splits
import os
import copy
import random
import argparse
import sys
from pdemodel.GNN_pde import GNNPDE_MLP
from pdemodel.classifier import Classifier
from data import get_train_val_test_split, even_quantile_labels
import json
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from sklearn import preprocessing as sk_prep
from pdemodel.best_params import best_params_dict
from prettytable import PrettyTable
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.io as sio


#
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)


# GCN
class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))
        # Linear transformation to match dimensions for skip connections
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, list):
            edge_index = torch.tensor(edge_index[0], dtype=torch.long)
        else:
            edge_index = edge_index.long()

        if edge_weight is not None:
            print(f"edge_weight dtype: {edge_weight.dtype}, shape: {edge_weight.shape}")

        z = x
        # z_1 = None  # To store the output of the first layer --- for analysis
        x_proj = self.input_proj(x)  # Project x to the same dimension as hidden_dim
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight=edge_weight)
            if i > 0:  # Add skip connection after the first layer
                z = z + x_proj
            z = self.activation(z)
            # z = F.dropout(z, p=self.dropout, training=self.training)
            # if i == 0:
            #     z_1 = z.clone()
        return z

# GAT

class GConvGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers, heads=1):
        super(GConvGAT, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()

        # First layer: input_dim → hidden_dim
        self.layers.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False # keep output dim = hidden_dim
            )
        )
        # Remaining layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False
                )
            )

        # Projection for skip connections
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, list):
            edge_index = torch.tensor(edge_index[0], dtype=torch.long)
        else:
            edge_index = edge_index.long()

        z = x
        x_proj = self.input_proj(x)

        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index)  # GATConv ignores edge_weight
            if i > 0:
                z = z + x_proj
            z = self.activation(z)

        return z

# GIN
class GConvGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConvGIN, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers.append(GINConv(nn=mlp))

        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                activation(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers.append(GINConv(nn=mlp))

        # Projection for skip connections
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, list):
            edge_index = torch.tensor(edge_index[0], dtype=torch.long, device=x.device)
        else:
            edge_index = edge_index.long().to(x.device)

        z = x
        x_proj = self.input_proj(x)

        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index)   # standard GINConv does not use edge_weight
            if i > 0:
                z = z + x_proj
            z = self.activation(z)

        return z

# node2vec

class Node2VecEncoder(nn.Module):
    def __init__(
        self,
        edge_index,
        num_nodes,
        hidden_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        p=1.0,
        q=1.0,
        num_negative_samples=1,
        sparse=True
    ):
        super(Node2VecEncoder, self).__init__()

        self.num_nodes = num_nodes
        self.model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=hidden_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            num_negative_samples=num_negative_samples,
            sparse=sparse
        )

    def forward(self, x=None, edge_index=None, edge_weight=None):
        node_ids = torch.arange(self.num_nodes, device=self.model.embedding.weight.device)
        z = self.model(node_ids)   # [num_nodes, hidden_dim]
        return z

# GraphSage

class GConvSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConvSAGE, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

        self.layers.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, list):
            edge_index = edge_index[0]
        edge_index = edge_index.long().to(x.device)

        z = x
        for i, conv in enumerate(self.layers):
            z_new = conv(z, edge_index)
            if i == 0:
                z = self.activation(z_new)
            else:
                z = self.activation(z_new + z)
        return z


class CustomNodePropPredDataset:
    def __init__(self, name, custom_num_classes):
        # Load the original dataset
        self.dataset = NodePropPredDataset(name=name)
        self._num_classes = custom_num_classes

        # Set up data (your Data object from previous steps)
        self.data = None  # Initialized as None and populated later

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_features(self):
        # Dynamically fetch the number of features from dataset
        graph = self.dataset.graph  # Assuming self.dataset.graph contains 'node_feat'
        return graph['node_feat'].shape[1]

    def __getitem__(self, idx):
        # Return the single data object (since this dataset contains only one graph)
        if idx != 0:
            raise IndexError("CustomNodePropPredDataset contains only one graph.")
        return self.data

    def __len__(self):
        # The length is 1 because it contains only one graph
        return 1

    # Forward any other attribute or method calls to the original dataset
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    ham_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def count_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb


def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim, proj_dim, num_features, a1, a2):
        super(Encoder, self).__init__()
        self.encoder1, self.encoder2 = encoder
        self.a1, self.a2 = a1, a2

    def forward(self, x, edge_index, edge_weight=None):
        z1 = self.encoder1(x, [edge_index, edge_weight])
        z2 = self.encoder2(x)
        return z1, z2

    def eval_test(self, x, edge_index, edge_weight=None):
        with torch.no_grad():
            z1 = self.encoder1(x, [edge_index, edge_weight])
            z2 = self.encoder2(x)
            z = self.a1 * z1 + self.a2 * z2
        return z

    def cal_reg(self, z1, z2):
        return self.encoder.odeblock.cal_loss(z1, z2)


def loss_fn_batch(x, y):
    # add batch dimension
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1).mean()
    return loss


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 1 - 1 * (x * y).sum(dim=-1).mean()
    cos_sim = (x * y).sum(dim=-1)
    cos_sim_var = cos_sim.var()
    return loss, cos_sim_var


def loss_fn_plus(x, y, lamba=0.1):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    pca_x = PCA()
    pca_y = PCA()
    pca_x.fit(x.detach().cpu().numpy())
    pca_y.fit(y.detach().cpu().numpy())
    main_feature_vector_x = pca_x.components_[0]
    main_feature_vector_y = pca_y.components_[0]
    main_feature_vector_x_tensor = torch.from_numpy(main_feature_vector_x).to(x.device)
    main_feature_vector_y_tensor = torch.from_numpy(main_feature_vector_y).to(y.device)
    loss_1 = 1 - 1 * (x * y).sum(dim=-1).mean()
    cos_sim = (x * y).sum(dim=-1)
    cos_sim_var = cos_sim.var()
    # Compute the magnitudes
    dot_product = torch.dot(main_feature_vector_x_tensor, main_feature_vector_y_tensor)
    magnitude_x = torch.norm(main_feature_vector_x_tensor)
    magnitude_y = torch.norm(main_feature_vector_y_tensor)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_x * magnitude_y)
    abs_cos_theta = torch.abs(cos_theta)
    loss_2 = abs_cos_theta
    loss = loss_1 + lamba * loss_2
    return loss, cos_sim_var


def vicreg_loss(z1, z2, lambd=1.0, mu=1.0, nu=1.0, gamma=1.0, epsilon=1e-4):
    # Number of nodes and feature dimensions
    N, d = z1.size()
    # Invariance component: Mean squared error between z1 and z2
    invariance = F.mse_loss(z1, z2, reduction='none')
    invariance_contributions = invariance.mean(dim=1)  # Per-node contributions

    # Variance component: Enforce non-collapse by ensuring variance along feature dimensions
    def variance_penalty(z):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + epsilon)
        return F.relu(gamma - std)

    variance_contributions = variance_penalty(z1).mean() + variance_penalty(z2).mean()

    # Covariance component: Decorrelate features
    def covariance_penalty(z):
        z = z - z.mean(dim=0)  # Center the features
        cov = (z.T @ z) / (N - 1)  # Covariance matrix
        off_diag = cov - torch.diag(torch.diag(cov))  # Zero the diagonal
        return off_diag.pow(2).sum(dim=1) / d  # Per-feature contributions

    covariance_contributions = covariance_penalty(z1).mean() + covariance_penalty(z2).mean()
    # Combine components
    loss = lambd * invariance.mean() + mu * variance_contributions + nu * covariance_contributions
    # Calculate variance of the loss contributions
    loss_contributions = (lambd * invariance_contributions).flatten()
    loss_var = loss_contributions.var()
    return loss, loss_var


def barlow_twin_loss(x, y, lambd=1.0):
    # Normalize the inputs
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    # Cross-correlation matrix computation
    C = torch.einsum('bi,bj->ij', x, y) / x.size(0)
    # Compute invariance term (diagonal close to 1)
    invariance_term = (1 - torch.diagonal(C)).pow(2)
    # Compute redundancy reduction term (off-diagonals close to 0)
    off_diag = C.clone()
    off_diag[torch.eye(C.size(0), device=C.device).bool()] = 0
    redundancy_term = off_diag.pow(2)
    # Combine the two terms
    loss = invariance_term.sum() + lambd * redundancy_term.sum()
    loss_var = invariance_term.var()
    return loss, loss_var


def loss_fn_cos(x1, y1):
    loss_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cossim = loss_cos(x1, y1)
    loss_cos_dis = 1 - loss_cossim
    loss_cos_dis_all = loss_cos_dis.sum()
    loss_var = loss_cos_dis.var()
    return loss_cos_dis_all, loss_var


def loss_fn_cos_plus(x1, y1, lamba=0.1):
    loss_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cossim = loss_cos(x1, y1)
    loss_cos_dis = 1 - loss_cossim
    loss_cos_dis_all = loss_cos_dis.sum()
    pca_x = PCA()
    pca_y = PCA()
    pca_x.fit(x1.detach().cpu().numpy())
    pca_y.fit(y1.detach().cpu().numpy())
    main_feature_vector_x = pca_x.components_[0]
    main_feature_vector_y = pca_y.components_[0]
    main_feature_vector_x_tensor = torch.from_numpy(main_feature_vector_x).to(x1.device)
    main_feature_vector_y_tensor = torch.from_numpy(main_feature_vector_y).to(y1.device)
    dot_product = torch.dot(main_feature_vector_x_tensor, main_feature_vector_y_tensor)
    magnitude_x = torch.norm(main_feature_vector_x_tensor)
    magnitude_y = torch.norm(main_feature_vector_y_tensor)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_x * magnitude_y)
    abs_cos_theta = torch.abs(cos_theta)
    loss_2 = abs_cos_theta
    loss_var = loss_cos_dis.var()
    loss = loss_cos_dis_all + lamba * loss_2
    return loss, loss_var


def loss_fn_euclidean(x1, y1):
    loss_euclidean = torch.nn.PairwiseDistance(p=2)
    loss_euclidean_dis = loss_euclidean(x1, y1)
    loss_euclidean_dis_all = loss_euclidean_dis.sum()
    loss_var = loss_euclidean_dis.var()
    return loss_euclidean_dis_all, loss_var


def loss_fn_euclidean_mean(x1, y1):
    loss_euclidean = torch.nn.PairwiseDistance(p=2)
    loss_euclidean_dis = loss_euclidean(x1, y1)
    loss_euclidean_dis_all = loss_euclidean_dis.mean()
    loss_var = loss_euclidean_dis.var()
    return loss_euclidean_dis_all, loss_var


def get_rows_by_class(feature_matrix, labels, target_class):
    selected_indices = np.where(labels == target_class)[0]
    return feature_matrix[selected_indices]


def train(encoder_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    if opt['loss'] == 'eucsum':
        loss, loss_var = loss_fn_euclidean(z1, z2)
    elif opt['loss'] == 'eucmean':
        loss, loss_var = loss_fn_euclidean_mean(z1, z2)
    elif opt['loss'] == 'cossum':
        loss, loss_var = loss_fn_cos(z1, z2)
    elif opt['loss'] == 'cosmean':
        loss, loss_var = loss_fn(z1, z2)
    elif opt['loss'] == 'barlow_twin':
        loss, loss_var = barlow_twin_loss(z1, z2, lambd=0.5)
    elif opt['loss'] == 'vicreg_loss':
        loss, loss_var = vicreg_loss(z1, z2, lambd=0.1, mu=0.15, nu=0.5, gamma=1.5, epsilon=1e-4)
    elif opt['loss'] == 'cosmean_plus':
        loss, loss_var = loss_fn_plus(z1, z2, lamba=opt['lambda'])
    elif opt['loss'] == 'cossum_plus':
        loss, loss_var = loss_fn_cos_plus(z1, z2, lamba=opt['lambda'])
    else:
        raise NotImplementedError

    loss.backward()
    optimizer.step()
    return loss.item(), loss_var.item(), z1, z2


def test(encoder_model, data, opt, class_time):
    encoder_model.eval()
    z = encoder_model.eval_test(data.x, data.edge_index, data.edge_attr)
    z = z.detach()
    if opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed', 'ogbn-arxiv']:
        split = from_predefined_split(data)
    elif opt['dataset'] in ['wikics']:
        split = from_predefined_split_wiki(data, class_time)
    elif opt['dataset'] in ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'actor', 'crocodile']:
        split = from_predefined_split_heter(data, class_time)
    elif opt['dataset'] in ['wiki-cooc', 'roman-empire', 'amazon-ratings', 'minesweeper', 'workers', 'questions',
                            'arxiv-year', 'chameleon-filtered', 'squirrel-filtered']:
        split = from_predefined_split_heterophilous(data, class_time)
    else:
        split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator(num_epochs=opt['classifier_epochs'], learning_rate=opt['classifier_lr'],
                         weight_decay=opt['classifier_decay'])(z, data.y, split)
    return result


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def test_classifier(encoder_model, data, classifier, classifier_optimizer, opt):
    encoder_model.eval()
    embeds = encoder_model.eval_test(data.x, data.edge_index, data.edge_attr).detach()
    if opt['dataset'] == 'ogbn-arxiv':
        embeds = embeds.squeeze(0)
    embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")
    embeds = torch.FloatTensor(embeds).to(opt['cuda'])
    dur = []
    best_acc, best_val_acc = 0, 0
    print('Testing Phase ==== Please Wait.')
    n_classifier_epochs = 5000
    for epoch in range(n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()
        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[data.train_mask], data.y[data.train_mask])
        loss.backward()
        classifier_optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_acc = evaluate(classifier, embeds, data.y, data.val_mask)
        if epoch > 500:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = evaluate(classifier, embeds, data.y, data.test_mask)
                if test_acc > best_acc:
                    best_acc = test_acc
    print("Valid Accuracy {:.4f}".format(best_val_acc))
    print("Test Accuracy {:.4f}".format(best_acc))
    return best_acc


def main(opt):
    set_seed(opt['seed'])
    device = torch.device('cuda:' + str(opt['cuda']) if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.expanduser('~'), 'datasets')
    if opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name=opt['dataset'], transform=T.NormalizeFeatures())
    elif opt['dataset'] in ['Computers', 'Photo']:
        dataset = Amazon(path, name=opt['dataset'])
    elif opt['dataset'] in ['Coauthorcs']:
        dataset = Coauthor(path, name='CS')
    elif opt['dataset'] in ['Coauthorphy']:
        dataset = Coauthor(path, name='Physics')
    elif opt['dataset'] in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name=opt['dataset'])
    elif opt['dataset'] in ['wikics']:
        dataset = WikiCS(path)
    elif opt['dataset'] in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, name=opt['dataset'])
    elif opt['dataset'] in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=f'datasets/', name=opt['dataset'], geom_gcn_preprocess=True)
    elif opt['dataset'] in ['crocodile']:
        dataset = WikipediaNetwork(root=f'dataset/', name=opt['dataset'], geom_gcn_preprocess=False)
    elif opt['dataset'] in ['actor']:
        path = os.path.join(path, opt['dataset'])
        dataset = Actor(path)
    elif opt['dataset'] in ['arxiv-year']:
        dataset = CustomNodePropPredDataset(name='ogbn-arxiv', custom_num_classes=5)
        split_dir = 'data/splits'
        split_idx = load_fixed_splits('arxiv-year', split_dir)
        for i in range(len(split_idx)):
            for key in split_idx[i]:
                if not torch.is_tensor(split_idx[i][key]):
                    split_idx[i][key] = torch.as_tensor(split_idx[i][key])
        # Extract train, valid, and test splits
        train_masks = []
        valid_masks = []
        test_masks = []
        for split in split_idx:
            train_masks.append(split['train'])
            valid_masks.append(split['valid'])
            test_masks.append(split['test'])
        # Stack masks to create the desired 2D tensor structure
        train_mask = torch.stack(train_masks)  # Each row represents a split's train mask
        valid_mask = torch.stack(valid_masks)  # Each row represents a split's valid mask
        test_mask = torch.stack(test_masks)
        graph = dataset.graph
        nclass = 5
        label = even_quantile_labels(
            graph['node_year'].flatten(), nclass, verbose=False)
        label = torch.tensor(label)
        node_features = torch.tensor(graph['node_feat'])
        edges = graph['edge_index']
        data = Data(
            x=node_features,
            edge_index=torch.LongTensor(edges),
            y=label,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=valid_mask
        )
        dataset.data = data

    if opt['dataset'] in ['Computers', 'Photo', 'CoauthorCS', 'CoauthorPHY']:
        dataset.data = get_train_val_test_split(opt['seed'], dataset.data,
                                                train_examples_per_class=20, val_examples_per_class=None,
                                                test_examples_per_class=None,
                                                train_size=None, val_size=500, test_size=1000)
    elif opt['dataset'] in ['ogbn-arxiv']:
        split_idx = dataset.get_idx_split()
        ei = to_undirected(dataset.data.edge_index)
        data = Data(
            x=dataset.data.x,
            edge_index=ei,
            y=dataset.data.y.T.squeeze(0),
            train_mask=split_idx['train'],
            test_mask=split_idx['test'],
            val_mask=split_idx['valid'])
        dataset.data = data
    elif opt['dataset'] in ['wiki-cooc', 'roman-empire', 'amazon-ratings', 'minesweeper', 'workers', 'questions',
                            'chameleon-filtered', 'squirrel-filtered']:
        dataset = MyOwnDataset(path, name=opt['dataset'])
        data = np.load(os.path.join('/home/zyn/datasets/HeterophilousDatasets/data', f'{opt["dataset"].replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])
        edges = edges.T

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        data = Data(
            x=node_features,
            edge_index=torch.LongTensor(edges),
            y=labels,
            train_mask=train_masks,
            test_mask=test_masks,
            val_mask=val_masks
        )

        dataset.data = data

    data = dataset[0].to(device)

    # feature_matrix = data.x
    #
    # idx = torch.rand_like(feature_matrix.float()).argsort(dim=1)
    #
    # feature_shuffled = torch.gather(feature_matrix, dim=1, index=idx)
    #
    # if args.feature_shuffled:
    #     data.x = feature_shuffled
    # else:
    #     data.x = feature_matrix

    n_classes = dataset.num_classes

    num_nodes = data.x.shape[0]
    opt['num_nodes'] = num_nodes
    opt_gconv2 = copy.deepcopy(opt)  # Create a copy for gconv2

    # gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=opt['hidden_dim'], activation=torch.nn.ReLU,
    #                num_layers=opt['GCN_layers_1']).to(device)

    # gconv1 = GConvGAT(input_dim=dataset.num_features, hidden_dim=opt['hidden_dim'], activation=torch.nn.ReLU,
    #                    num_layers=opt['GAT_layers_1'], heads=opt['num_head']).to(device)

    gconv1 = GConvGIN(input_dim=dataset.num_features, hidden_dim=opt['hidden_dim'], activation=torch.nn.ReLU,
                     num_layers=opt['GIN_layers_1']).to(device)
    # gconv1 = GConvGAT(edge_index=data.edge_index, num_nodes=data.num_nodes, hidden_dim=opt['hidden_dim']).to(device)

    gconv2 = GNNPDE_MLP(opt_gconv2, dataset.num_features, device)

    count_parameters(gconv1)
    count_model_size(gconv1)
    count_parameters(gconv2)
    count_model_size(gconv2)

    encoder_model = Encoder(encoder=(gconv1, gconv2), hidden_dim=opt['hidden_dim'], proj_dim=opt['proj_dim'],
                            num_features=dataset.num_features, a1=args.a1, a2=args.a2).to(device)

    parameters = [
        {'params': p, 'name': n} for n, p in encoder_model.named_parameters() if p.requires_grad
    ]
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            print(f"Parameter Name: {param_group['name']}, Shape: {param.shape}")

    cnt_wait = 0
    best_loss = 1e9
    best_t = 0
    classifier = Classifier(opt['hidden_dim'], n_classes).to(device)
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                             lr=opt['classifier_lr'],
                                             weight_decay=opt['classifier_decay'])

    with tqdm(total=opt['epoch'], desc='(T)') as pbar:
        for epoch in range(1, opt['epoch'] + 1):
            loss, cos_sim_var, z1, z2 = train(encoder_model, data, optimizer)
            if loss < best_loss:
                best_loss = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(encoder_model.state_dict(),
                           'pkl/model_gin_linear_' + str(opt['GIN_layers_1']) + '_' + str(opt['hidden_dim']) + '_' +
                           opt['dataset'] + '.pkl')

                # torch.save(encoder_model.state_dict(),
                #            'pkl/model_n2v_linear_' + str(opt['hidden_dim']) + '_' +
                #            opt['dataset'] + '.pkl')
            else:
                cnt_wait += 1
            pbar.set_postfix({'loss': loss, 'cos_var': cos_sim_var})
            pbar.update()
            if cnt_wait == opt['patience']:
                print('Early stopping!')
                break
    results_lg = []
    encoder_model.load_state_dict(torch.load(
        'pkl/model_gin_linear_' + str(opt['GIN_layers_1']) + '_' + str(opt['hidden_dim']) + '_' + opt[
            'dataset'] + '.pkl', map_location='cpu'))

    # encoder_model.load_state_dict(torch.load(
    #     'pkl/model_n2v_linear_' + str(opt['hidden_dim']) + '_' + opt[
    #         'dataset'] + '.pkl', map_location='cpu'))

    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    classifier_seed = opt['classifier_seed']
    if opt['classifier'] == 'linear':

        for _ in range(opt['classifier_time']):
            set_seed(seed=classifier_seed)
            test_result = test_classifier(encoder_model, data, classifier, classifier_optimizer, opt)
            results_lg.append(test_result)
            classifier_seed += 1
    else:

        for class_time in range(opt['classifier_time']):
            set_seed(seed=classifier_seed)
            test_result = test(encoder_model, data, opt, class_time)
            classifier_seed += 1
            # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
            results_lg.append(test_result['acc'])

    test_mean = np.mean(np.array(results_lg))
    test_std = np.std(np.array(results_lg))
    print(f'(E): Best test_mean={test_mean:.4f}, std={test_std:.4f}')
    best_opt = opt
    return test_mean, test_std, best_opt


def merge_cmd_args(cmd_opt, opt):
    if cmd_opt['beltrami']:
        opt['beltrami'] = True
    if cmd_opt['function'] is not None:
        opt['function'] = cmd_opt['function']
    if cmd_opt['block'] is not None:
        opt['block'] = cmd_opt['block']
    if cmd_opt['attention_type'] != 'scaled_dot':
        opt['attention_type'] = cmd_opt['attention_type']
    if cmd_opt['self_loop_weight'] is not None:
        opt['self_loop_weight'] = cmd_opt['self_loop_weight']
    if cmd_opt['method'] is not None:
        opt['method'] = cmd_opt['method']
    if cmd_opt['step_size'] != 1:
        opt['step_size'] = cmd_opt['step_size']
    if cmd_opt['time'] is not None:
        opt['time'] = cmd_opt['time']
    if cmd_opt['epoch'] != 100:
        opt['epoch'] = cmd_opt['epoch']
    if not cmd_opt['not_lcc']:
        opt['not_lcc'] = False
    if cmd_opt['num_splits'] != 1:
        opt['num_splits'] = cmd_opt['num_splits']
    if cmd_opt['dropout'] is not None:
        opt['dropout'] = cmd_opt['dropout']
    if cmd_opt['hidden_dim'] is not None:
        opt['hidden_dim'] = cmd_opt['hidden_dim']
    if cmd_opt['decay'] is not None:
        opt['decay'] = cmd_opt['decay']
    if cmd_opt['self_loop_weight'] is not None:
        opt['self_loop_weight'] = cmd_opt['self_loop_weight']
    if cmd_opt['edge_homo'] != 0:
        opt['edge_homo'] = cmd_opt['edge_homo']

    if cmd_opt['lr'] is not None:
        opt['lr'] = cmd_opt['lr']
    if cmd_opt['input_dropout'] is not None:
        opt['input_dropout'] = cmd_opt['input_dropout']
    if cmd_opt['heads'] is not None:
        opt['heads'] = cmd_opt['heads']


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings("ignore")

    # setting arguments
    parser = argparse.ArgumentParser('GGD')
    parser.add_argument('--classifier_epochs', type=int, default=1000, help='classifier epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--patience', type=int, default=200, help='Patience')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--drop_prob', type=float, default=0.0, help='Tau value')
    parser.add_argument('--dataset', type=str, default='wikics', help='Dataset name: cora, citeseer, pubmed, cs, phy')
    parser.add_argument('--num_hop', type=int, default=0, help='graph power')
    parser.add_argument('--cuda', type=int, default=2, help='cuda')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--classifier_lr', type=float, default=0.005, help='classifier_lr.')
    parser.add_argument('--classifier_decay', type=float, default=0.0005, help='classifier_decay.')
    parser.add_argument('--classifier_seed', type=int, default=10, help='classifier_seed.')
    parser.add_argument('--classifier_time', type=int, default=2, help='classifier_time.')
    parser.add_argument('--classifier', type=str, default=None, help='classifier_time.')

    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension.')
    parser.add_argument('--proj_dim', type=int, default=256, help='proj_dim dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.4, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')

    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=3.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, default='euler',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint, predictor")
    parser.add_argument('--step_size', type=float, default=1.0,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")

    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')

    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=16,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    parser.add_argument('--data_norm', type=str, default='gcn',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')

    parser.add_argument('--runtime', type=int, default=1, help='run time')
    parser.add_argument('--loss', type=str, default='cossum', help='loss function,cosmean,cossum,eucmean,eucsum')
    parser.add_argument('--random_splits', dest='random_splits', action='store_true',
                        help='random_splits')

    # FDE args
    parser.add_argument('--alpha_ode', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_ode_1', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_ode_2', type=float, default=0.1, help='Factor in front matrix A.')
    # parser.add_argument('--fractional', action='store_true', help="Use fractional GNNs")
    parser.add_argument('--a1', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a2', type=float, default=0.5, help='weighted of FDE')

    # gread args
    parser.add_argument('--reaction_term', type=str, default='bspm', help='bspm, fisher, allen-cahn')
    parser.add_argument('--beta_diag', type=eval, default=False)

    # loss function regularization

    parser.add_argument('--lambda', type=float, default=0.1, help='regularization parameter')
    # GCN args
    parser.add_argument('--GCN_layers_1', type=int, default=2, help='The number of GCN layers')
    parser.add_argument('--GCN_layers_2', type=int, default=2, help='The number of GCN layers')

    # control feature shuffling
    parser.add_argument('--feature_shuffled', action='store_true', help='enable feature shuffling')

    # GAT args
    parser.add_argument('--GAT_layers_1', type=int, default=2, help='The number of GCN layers')
    parser.add_argument('--GAT_layers_2', type=int, default=2, help='The number of GCN layers')
    parser.add_argument('--num_head', type = int, default=2, help='The number of heads')

    # GIN args
    parser.add_argument('--GIN_layers_1', type=int, default=2, help='The number of GIN layers')
    parser.add_argument('--GIN_layers_2', type=int, default=2, help='The number of GIN layers')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    cmd_opt = vars(args)
    try:
        best_opt = best_params_dict[cmd_opt['dataset']]
        opt = {**cmd_opt, **best_opt}
        merge_cmd_args(cmd_opt, opt)
    except KeyError:
        opt = cmd_opt
    #
    # opt = cmd_opt
    test_result = []

    timestr = time.strftime("%H%M%S")
    # create a folder to store the log
    if not os.path.exists("log"):
        os.makedirs("log")
    if not os.path.exists("log_gin"):
        os.makedirs("log_gin")
    if not os.path.exists("log_n2v"):
        os.makedirs("log_n2v")
    filename = "log_gin/" + str(args.dataset) + "_" + timestr + ".txt"
    command_args = " ".join(sys.argv)
    with open(filename, 'a') as f:
        json.dump(command_args, f)
        f.write("\n")

    for _ in range(args.runtime):
        test_acc, test_std, opt_final = main(opt)
        test_result.append(test_acc)
        with open(filename, 'a') as f:
            f.write("test_acc: " + str(test_acc) + "\n")
            f.write("test_std: " + str(test_std) + "\n")

    acc_mean = np.mean(np.array(test_result))
    acc_std = np.std(np.array(test_result))
    print(f'(Final): Best test mean={acc_mean:.4f}, std={acc_std:.4f}')
    with open(filename, 'a') as f:
        f.write("test_acc_mean: " + str(acc_mean) + "\n")
        f.write("test_acc_std: " + str(acc_std) + "\n")
        json.dump(opt_final, f, indent=2)
        f.write("\n")
    # change saved filename to the include the best test acc
    os.rename(filename, filename[:-4] + str(acc_mean) + ".txt")


