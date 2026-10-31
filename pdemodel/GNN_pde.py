import torch
from torch import nn
import torch.nn.functional as F
from pdemodel.utilspde import Meter
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
from pdemodel.block_constant_fractional_adap import ConstantODEblock_FRAC_adap
from pdemodel.block_alpha_param import ConstantODEblock_FRAC_alpha
from pdemodel.block_transformer_fractional import AttODEblock_FRAC
from pdemodel.function_gread import ODEFuncGread
from pdemodel.function_laplacian_graphcon import LaplacianODEFunc_graphcon



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
    # self.m2 = nn.Linear(opt['hidden_dim'], self.num_classes)
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
class GNNPDE(BaseGNN):
  def __init__(self, opt,num_features, data, device=torch.device('cpu')):
    super(GNNPDE, self).__init__(opt, num_features,device)
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
    elif opt['function'] == 'graphcon':
        self.f = LaplacianODEFunc_graphcon
    else:
        raise ValueError('Function not implemented')
    if opt['block'] == 'constant':
        block = ConstantODEblock
    # add constant_frac option
    elif opt['block'] == 'constant_frac':
        block = ConstantODEblock_FRAC
    # add Att_frac option
    elif opt['block'] == 'attention_frac':
        block = AttODEblock_FRAC
    elif opt['block'] == 'attention':
        block = AttODEblock
    elif opt['block'] == 'constantbatch':
        block = ConstantODEblockbatch
    # add constant_frac_adaptive option
    elif opt['block'] == 'constant_frac_adap':
        block = ConstantODEblock_FRAC_adap
    else:
        raise ValueError('Block not implemented')
    # block = ConstantODEblock
    # block = AttODEblock

    # self.regularization_fns = None

    time_tensor = torch.tensor([0, self.T]).to(device)
    # if not self.training:
    #   print('time_tensor',time_tensor)
    self.odeblock = block(self.f, opt, data, device, t=time_tensor).to(device)
    # self.prelu = nn.PReLU()
    # self.bn = nn.BatchNorm1d(opt['hidden_dim'])

  def forward(self, x, adj):
    # Encode each node based on its feature.




    # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    # Linear encoder to generate z0
    x = self.m1(x)

    # if self.opt['use_mlp']:
    #   x = F.dropout(x, self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    #
    #
    #
    if self.opt['batch_norm']:
      z0 = self.bn_in(x)
      # x = self.bn_in(x)
    # x = self.bn_in(x)
    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      z0 = torch.cat([x, c_aux], dim=1)

    # graphcon
    if 'graphcon' in self.opt['function']:
      x_init = z0.clone()
      x = torch.cat([z0, x_init], dim=-1)
      self.odeblock.set_x0(x)
      z = self.odeblock(x,adj)
      z = z[:,self.opt['hidden_dim']:]
    else:
      self.odeblock.set_x0(z0)
      if 'constant_frac_adap' in self.opt['block']:
        z, alpha = self.odeblock(z0, adj)
      else:
        z = self.odeblock(z0, adj)

    # if self.opt['augment']:
    #   z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # if self.opt['fractional']:
    #   temp_time = self.opt['time']
    #   temp_method = self.opt['method']
    #   temp_step_size = self.opt['step_size']
    #
    #   self.opt['time'] = 1 # self.opt['fa_layer_time'] #1.0
    #   self.opt['method'] = 'rk4' # self.opt['fa_layer_method']#'rk4'
    #   self.opt['step_size'] = 1 # self.opt['fa_layer_step_size']#1.0
    #   self.odeblock.set_x0(z)
    #   # self.odeblock.odefunc.edge_index = add_edges(self, self.opt)
    #   # if self.opt['edge_sampling_rmv'] != 0:
    #   #   edge_sampling(self, z, self.opt)
    #
    #   z = self.odeblock(z)
    #   self.odeblock.odefunc.edge_index = self.data_edge_index
    #
    #   self.opt['time'] = temp_time
    #   self.opt['method'] = temp_method
    #   self.opt['step_size'] = temp_step_size

    # Activation.
    z = F.relu(z)
    # z = self.prelu(z)
    #
    # if self.opt['fc_out']:
    #   z = self.fc(z)
    #   z = F.relu(z)
    #
    # Dropout.
    # z = F.dropout(z, self.opt['dropout'], training=self.training)
    #
    # # Decode each node embedding to get node label.
    # z = self.m2(z)
    # z = self.bn(z)
    if 'constant_frac_adap' in self.opt['block']:
      return z0, z, alpha
    else:
      return z

# Define GNN model without ODE/FDE block

class GNNPDE_MLP(BaseGNN):
  def __init__(self, opt, num_features, device=torch.device('cpu')):
    super(GNNPDE_MLP, self).__init__(opt, num_features,device)
    # self.f = set_function(opt)
    # block = set_block(opt)
    # if opt['function'] == 'laplacian':
    #     self.f = LaplacianODEFunc
    # elif opt['function'] == 'GAT':
    #     self.f = ODEFuncAtt
    # elif opt['function'] == 'beltrans':
    #     self.f = ODEFuncBektramiAtt
    # elif opt['function'] == 'bellap':
    #     self.f = ODEFuncBektramiLAP
    # elif opt['function'] == 'transformer':
    #     self.f = ODEFuncTransformerAtt
    # elif opt['function'] == 'GATnorm':
    #     self.f = ODEFuncAttNorm
    # elif opt['function'] == 'lapconv':
    #     self.f = ODEFuncLapCONV
    # elif opt['function'] == 'gread':
    #     self.f = ODEFuncGread
    # elif opt['function'] == 'graphcon':
    #     self.f = LaplacianODEFunc_graphcon
    # else:
    #     raise ValueError('Function not implemented')
    # if opt['block'] == 'constant':
    #     block = ConstantODEblock
    # # add constant_frac option
    # elif opt['block'] == 'constant_frac':
    #     block = ConstantODEblock_FRAC
    # # add Att_frac option
    # elif opt['block'] == 'attention_frac':
    #     block = AttODEblock_FRAC
    # elif opt['block'] == 'attention':
    #     block = AttODEblock
    # elif opt['block'] == 'constantbatch':
    #     block = ConstantODEblockbatch
    # else:
    #     raise ValueError('Block not implemented')


    # block = ConstantODEblock
    # block = AttODEblock

    # self.regularization_fns = None

    # time_tensor = torch.tensor([0, self.T]).to(device)
    # if not self.training:
    #   print('time_tensor',time_tensor)
    # self.odeblock = block(self.f, opt, data, device, t=time_tensor).to(device)
    # self.prelu = nn.PReLU()
    # self.bn = nn.BatchNorm1d(opt['hidden_dim'])

  def forward(self, x):
    # Encode each node based on its feature.



    # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    # Linear encoder to generate z0
    x = self.m1(x)

    # if self.opt['use_mlp']:
    #   x = F.dropout(x, self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    #
    #
    #
    if self.opt['batch_norm']:
      z0 = self.bn_in(x)       # x = self.bn_in(x)
    # x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    # if self.opt['augment']:
    #   c_aux = torch.zeros(x.shape).to(self.device)
    #   z0 = torch.cat([x, c_aux], dim=1)
    #
    # # graphcon
    # if 'graphcon' in self.opt['function']:
    #   x_init = z0.clone()
    #   x = torch.cat([z0, x_init], dim=-1)
    #   self.odeblock.set_x0(x)
    #   z = self.odeblock(x,adj)
    #   z = z[:,self.opt['hidden_dim']:]
    # else:
    #   self.odeblock.set_x0(z0)
    #   z = self.odeblock(z0, adj)

    # if self.opt['augment']:
    #   z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # if self.opt['fractional']:
    #   temp_time = self.opt['time']
    #   temp_method = self.opt['method']
    #   temp_step_size = self.opt['step_size']
    #
    #   self.opt['time'] = 1 # self.opt['fa_layer_time'] #1.0
    #   self.opt['method'] = 'rk4' # self.opt['fa_layer_method']#'rk4'
    #   self.opt['step_size'] = 1 # self.opt['fa_layer_step_size']#1.0
    #   self.odeblock.set_x0(z)
    #   # self.odeblock.odefunc.edge_index = add_edges(self, self.opt)
    #   # if self.opt['edge_sampling_rmv'] != 0:
    #   #   edge_sampling(self, z, self.opt)
    #
    #   z = self.odeblock(z)
    #   self.odeblock.odefunc.edge_index = self.data_edge_index
    #
    #   self.opt['time'] = temp_time
    #   self.opt['method'] = temp_method
    #   self.opt['step_size'] = temp_step_size

    # Activation.
    z = F.relu(z0)
    # z = self.prelu(z)
    #
    # if self.opt['fc_out']:
    #   z = self.fc(z)
    #   z = F.relu(z)
    #
    # Dropout.
    # z = F.dropout(z, self.opt['dropout'], training=self.training)
    #
    # # Decode each node embedding to get node label.
    # z = self.m2(z)
    # z = self.bn(z)
    return z



# --------------------------------------------------------------------------------------------
class GNN(BaseGNN):
  def __init__(self, opt, num_features, data, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, num_features, device)
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
      # add constant_frac option
    elif opt['block'] == 'constant_frac':
      block = ConstantODEblock_FRAC
      # add Att_frac option
    elif opt['block'] == 'attention_frac':
      block = AttODEblock_FRAC
    elif opt['block'] == 'attention':
      block = AttODEblock
    elif opt['block'] == 'constantbatch':
      block = ConstantODEblockbatch
    else:
      raise ValueError('Block not implemented')
      # block = ConstantODEblock
      # block = AttODEblock

      # self.regularization_fns = None

    time_tensor = torch.tensor([0, self.T]).to(device)
    # if not self.training:
    #   print('time_tensor',time_tensor)
    self.odeblock = block(self.f, opt, data, device, t=time_tensor).to(device)
    # self.prelu = nn.PReLU()
    # self.bn = nn.BatchNorm1d(opt['hidden_dim'])

  def forward(self, x, adj):
      # Encode each node based on its feature.

      # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      # Linear encoder to generate z0
    x = self.m1(x)

    # if self.opt['use_mlp']:
    #   x = F.dropout(x, self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
    #   x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    #
    #
    #
    if self.opt['batch_norm']:
      z0 = self.bn_in(x)  # x = self.bn_in(x)
      # x = self.bn_in(x)
      # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      z0 = torch.cat([x, c_aux], dim=1)

    # self.odeblock.set_x0(z0)
    #
    # z = self.odeblock(z0, adj)

    # if self.opt['augment']:
    #   z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # if self.opt['fractional']:
    #   temp_time = self.opt['time']
    #   temp_method = self.opt['method']
    #   temp_step_size = self.opt['step_size']
    #
    #   self.opt['time'] = 1 # self.opt['fa_layer_time'] #1.0
    #   self.opt['method'] = 'rk4' # self.opt['fa_layer_method']#'rk4'
    #   self.opt['step_size'] = 1 # self.opt['fa_layer_step_size']#1.0
    #   self.odeblock.set_x0(z)
    #   # self.odeblock.odefunc.edge_index = add_edges(self, self.opt)
    #   # if self.opt['edge_sampling_rmv'] != 0:
    #   #   edge_sampling(self, z, self.opt)
    #
    #   z = self.odeblock(z)
    #   self.odeblock.odefunc.edge_index = self.data_edge_index
    #
    #   self.opt['time'] = temp_time
    #   self.opt['method'] = temp_method
    #   self.opt['step_size'] = temp_step_size

    # Activation.
    z = F.relu(z0)
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
    return z0, z

  # def cal_loss(self,x1,x2):
  #   return self.f.cal_loss(x1,x2)
  #   print("self.f",self.f)
  #
  #   h_1 = self.f.forward_loss(0,x1)
  #   h_2 = self.f.forward_loss(0,x2)
  #   return (torch.abs(torch.mean(h_1-h_2)))
