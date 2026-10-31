import torch
from torch import nn
import torch.nn.functional as F
from .utilspde import Meter
# from model_configurations import set_block, set_function
from torch_geometric.nn.conv import MessagePassing
from .function_GAT_attention import ODEFuncAtt
from .block_constant import ConstantODEblock, AttODEblock
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from pdemodel.function_GAT_attention import ODEFuncAtt
from pdemodel.block_constant import ConstantODEblock, AttODEblock
from pdemodel.function_laplacian_diffusion import LaplacianODEFunc
from pdemodel.function_beltrami_trans import ODEFuncBektramiAtt
from pdemodel.function_beltrami_lap import ODEFuncBektramiLAP
from pdemodel.function_transformer_attention import ODEFuncTransformerAtt
from pdemodel.function_GAT_norm import ODEFuncAttNorm
from pdemodel.function_laplacian_convection import ODEFuncLapCONV
from pdemodel.block_constant_batch import ConstantODEblockbatch
from pdemodel.block_constant_fractional import ConstantODEblock_FRAC
from pdemodel.block_transformer_fractional import AttODEblock_FRAC
from pdemodel.function_gread import ODEFuncGread
class BaseGNN(MessagePassing):
  def __init__(self, opt,num_features, device=torch.device('cpu')):
    super(BaseGNN, self).__init__()
    self.opt = opt
    self.T = opt['time']
    self.num_classes = 2
    self.num_features = num_features
    # self.num_nodes = num_nodes
    self.device = device
    self.fm = Meter()
    self.bm = Meter()

    # if opt['beltrami']:
    #   self.mx = nn.Linear(self.num_features, opt['feat_hidden_dim'])
    #   self.mp = nn.Linear(opt['pos_enc_dim'], opt['pos_enc_hidden_dim'])
    #   opt['hidden_dim'] = opt['feat_hidden_dim'] + opt['pos_enc_hidden_dim']
    # else:
    #   self.m1 = nn.Linear(self.num_features, opt['hidden_dim'])


    self.m1 = nn.Linear(self.num_features, opt['hidden_dim'])
    if self.opt['use_mlp']:
      self.m11 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
      self.m12 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    # if opt['use_labels']:
    #   # todo - fastest way to propagate this everywhere, but error prone - refactor later
    #   opt['hidden_dim'] = opt['hidden_dim'] + self.num_classes
    # else:
    #   self.hidden_dim = opt['hidden_dim']
    self.hidden_dim = opt['hidden_dim']
    if opt['fc_out']:
      self.fc = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    self.m2 = nn.Linear(opt['hidden_dim'], self.num_classes)
    if self.opt['batch_norm']:
      self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'])
      self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])
    # self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'])
    # self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])
    # self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)

  def getNFE(self):
    return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe

  def resetNFE(self):
    self.odeblock.odefunc.nfe = 0
    self.odeblock.reg_odefunc.odefunc.nfe = 0

  def reset(self):
    self.m1.reset_parameters()
    self.m2.reset_parameters()

  def __repr__(self):
    return self.__class__.__name__

# Define the GNN model.
class GNNPDE_GRAPH(BaseGNN):
  def __init__(self, opt, num_features, data, device=torch.device('cpu')):
    super(GNNPDE_GRAPH, self).__init__(opt,num_features, device)
    # self.f = set_function(opt)
    # block = set_block(opt)
    if opt['function'] == 'laplacian':
      self.f = LaplacianODEFunc
    elif opt['function'] == 'GAT':
      self.f = ODEFuncAtt
    elif opt['function'] == 'beltrans':
      self.f = ODEFuncBektramiAtt
    elif opt['function'] == 'bellap':
      self.f = ODEFuncBektramiLAP
    elif opt['function'] == 'transformer':
      self.f = ODEFuncTransformerAtt
    elif opt['function'] == 'GATnorm':
      self.f = ODEFuncAttNorm
    elif opt['function'] == 'lapconv':
      self.f = ODEFuncLapCONV
    elif opt['function'] == 'gread':
      self.f = ODEFuncGread
    else:
      raise ValueError('Function not implemented')
    if opt['block'] == 'constant':
      block = ConstantODEblock
    elif opt['block'] == 'constant_frac':
      block = ConstantODEblock_FRAC
    elif opt['block'] == 'attention_frac':
      block = AttODEblock_FRAC
    elif opt['block'] == 'attention':
      block = AttODEblock
    elif opt['block'] == 'constantbatch':
      block = ConstantODEblockbatch
    else:
      raise ValueError('Block not implemented')

    # self.regularization_fns = None

    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, opt, data, device, t=time_tensor).to(device)
    # self.prelu = nn.PReLU()
    # self.bn = nn.BatchNorm1d(opt['hidden_dim'])

  def forward(self, x, adj, batch):
    # Encode each node based on its feature.



    # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)

    # if self.opt['use_mlp']:
    #   x = F.dropout(x, self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    if self.opt['batch_norm']:
      x = self.bn_in(x)


    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    self.odeblock.set_x0(x)

    # print('x0 shape: ',x.shape)
    # print('adj shape: ',adj[0].shape)
    z = self.odeblock(x, adj)

    # if self.opt['augment']:
    #   z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)
    # z = self.prelu(z)
    #
    # if self.opt['fc_out']:
    #   z = self.fc(z)
    #   z = F.relu(z)
    #
    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)
    #
    # # Decode each node embedding to get node label.
    # z = self.m2(z)
    # z = self.bn(z)
    # z = global_add_pool(z, batch)
    return z
