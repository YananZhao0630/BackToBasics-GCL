import torch
from torch import nn
from .utilspde import get_rw_adj, gcn_norm_fill_val
import numpy as np
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops

class ODEblock(nn.Module):
  def __init__(self, odefunc, regularization_fns, opt,  device, t):
    super(ODEblock, self).__init__()
    self.opt = opt
    self.t = t

    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, device)

    # self.nreg = len(regularization_fns)
    # self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = None
    self.set_tol()

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()
    # self.reg_odefunc.odefunc.x0 = x0.clone().detach()

  def set_tol(self):
    self.atol = self.opt['tol_scale'] * 1e-7
    self.rtol = self.opt['tol_scale'] * 1e-9
    if self.opt['adjoint']:
      self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
      self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"


class ConstantODEblockbatch(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, device, t=torch.tensor([0, 1])):
    super(ConstantODEblockbatch, self).__init__(odefunc, regularization_fns, opt, device, t)


    self.aug_dim = 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, device)
    # if opt['data_norm'] == 'rw':
    #   edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
    #                                                                fill_value=opt['self_loop_weight'],
    #                                                                num_nodes=data.num_nodes,
    #                                                                dtype=data.x.dtype)
    # else:
    #   edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
    #                                        fill_value=opt['self_loop_weight'],
    #                                        num_nodes=data.num_nodes,
    #                                        dtype=data.x.dtype)
    # self.odefunc.edge_index = edge_index.to(device)
    # self.odefunc.edge_weight = edge_weight.to(device)
    # self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    self.device = device
    self.opt = opt

  def forward(self, x, adj):
    self.t = torch.linspace(0, 10, 10)
    # self.t = torch.linspace(0, 5, 20)
    t = self.t.type_as(x)

    # self.odefunc.edge_index = adj._indices().to(self.device)
    # self.odefunc.edge_attr = adj._values().to(self.device)

    # self.edge_index = adj._indices().to(self.device)
    # self.edge_attr = adj._values().to(self.device)

    self.edge_index = adj[0]
    self.edge_attr = adj[1]

    if self.opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(self.edge_index, edge_weight=self.edge_attr, norm_dim=1,
                                                                   fill_value=self.opt['self_loop_weight'],
                                                                   num_nodes=self.opt['num_nodes'],
                                                                   dtype=x.dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(self.edge_index, edge_weight=self.edge_attr,
                                           fill_value=self.opt['self_loop_weight'],
                                           num_nodes=self.opt['num_nodes'],
                                           dtype=x.dtype)

    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)

    integrator = self.train_integrator if self.training else self.test_integrator




    func = self.odefunc
    state = x

    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options=dict(step_size = self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], ),
        atol=self.atol,
        rtol=self.rtol)
    # z = state_dt[1]
    if self.training:
      z = state_dt
    else:
      z = state_dt[-1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"



class AttODEblock(ODEblock):
  def __init__(self,  odefunc, regularization_fns, opt, device, t=torch.tensor([0, 1]), gamma=0.5):
    super(AttODEblock, self).__init__(odefunc, regularization_fns, opt, device, t)

    self.aug_dim = 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, device)
    # self.odefunc.edge_index, self.odefunc.edge_weight = data.edge_index, edge_weight=data.edge_attr
    # edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
    #                                      fill_value=opt['self_loop_weight'],
    #                                      num_nodes=data.num_nodes,
    #                                      dtype=data.x.dtype)
    # self.odefunc.edge_index = edge_index.to(device)
    # self.odefunc.edge_weight = edge_weight.to(device)
    # self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    self.device = device
    self.opt = opt
    # parameter trading off between attention and the Laplacian
    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,
                                                          device, edge_weights=self.odefunc.edge_weight).to(device)

  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention


  def forward(self, x, adj):

    # self.t = torch.linspace(0, 5, 10)
    t = self.t.type_as(x)

    # self.odefunc.edge_index = adj._indices().to(self.device)
    # self.odefunc.edge_attr = adj._values().to(self.device)

    # self.edge_index = adj._indices().to(self.device)
    # self.edge_attr = adj._values().to(self.device)

    self.edge_index = adj[0]
    self.edge_attr = adj[1]

    if self.opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(self.edge_index, edge_weight=self.edge_attr, norm_dim=1,
                                                                   fill_value=self.opt['self_loop_weight'],
                                                                   num_nodes=self.opt['num_nodes'],
                                                                   dtype=x.dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(self.edge_index, edge_weight=self.edge_attr,
                                           fill_value=self.opt['self_loop_weight'],
                                           num_nodes=self.opt['num_nodes'],
                                           dtype=x.dtype)
    if self.opt['self_loop_weight'] > 0:
      edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                                   fill_value=self.opt['self_loop_weight'])
    else:
      edge_index, edge_weight = edge_index, edge_weight

    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)
    self.odefunc.attention_weights = self.get_attention_weights(x)
    # self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights
    integrator = self.train_integrator if self.training else self.test_integrator

    # reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))

    # func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    # state = (x,) + reg_states if self.training and self.nreg > 0 else x
    func = self.odefunc
    state = x

    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options={'step_size': self.opt['adjoint_step_size']},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        atol=self.atol,
        rtol=self.rtol)

    # if self.training and self.nreg > 0:
    #   z = state_dt[0][1]
    #   reg_states = tuple(st[1] for st in state_dt[1:])
    #   return z, reg_states
    # else:
    #   z = state_dt[1]
    z = state_dt[1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"




class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])
    self.edge_weights = edge_weights

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h

    # if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
    #   self.output_var_x = nn.Parameter(torch.ones(1))
    #   self.lengthscale_x = nn.Parameter(torch.ones(1))
    #   self.output_var_p = nn.Parameter(torch.ones(1))
    #   self.lengthscale_p = nn.Parameter(torch.ones(1))
    #   self.Qx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
    #   self.init_weights(self.Qx)
    #   self.Vx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
    #   self.init_weights(self.Vx)
    #   self.Kx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
    #   self.init_weights(self.Kx)
    #
    #   self.Qp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
    #   self.init_weights(self.Qp)
    #   self.Vp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
    #   self.init_weights(self.Vp)
    #   self.Kp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
    #   self.init_weights(self.Kp)
    #
    # else:
    #   if self.opt['attention_type'] == "exp_kernel":
    #     self.output_var = nn.Parameter(torch.ones(1))
    #     self.lengthscale = nn.Parameter(torch.ones(1))

    self.Q = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.Q)

    self.V = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.V)

    self.K = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.K)

    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, x, edge):
    """
    x might be [features, augmentation, positional encoding, labels]
    """
    # if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
    self.opt['beltrami'] = 0
    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      p = x[:, self.opt['feat_hidden_dim']: label_index]
      x = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)

      qx = self.Qx(x)
      kx = self.Kx(x)
      vx = self.Vx(x)
      # perform linear operation and split into h heads
      kx = kx.view(-1, self.h, self.d_k)
      qx = qx.view(-1, self.h, self.d_k)
      vx = vx.view(-1, self.h, self.d_k)
      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      kx = kx.transpose(1, 2)
      qx = qx.transpose(1, 2)
      vx = vx.transpose(1, 2)
      src_x = qx[edge[0, :], :, :]
      dst_x = kx[edge[1, :], :, :]

      qp = self.Qp(p)
      kp = self.Kp(p)
      vp = self.Vp(p)
      # perform linear operation and split into h heads
      kp = kp.view(-1, self.h, self.d_k)
      qp = qp.view(-1, self.h, self.d_k)
      vp = vp.view(-1, self.h, self.d_k)
      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      kp = kp.transpose(1, 2)
      qp = qp.transpose(1, 2)
      vp = vp.transpose(1, 2)
      src_p = qp[edge[0, :], :, :]
      dst_p = kp[edge[1, :], :, :]

      prods = self.output_var_x ** 2 * torch.exp(
        -torch.sum((src_x - dst_x) ** 2, dim=1) / (2 * self.lengthscale_x ** 2)) \
              * self.output_var_p ** 2 * torch.exp(
        -torch.sum((src_p - dst_p) ** 2, dim=1) / (2 * self.lengthscale_p ** 2))

      v = None

    else:
      q = self.Q(x)
      k = self.K(x)
      v = self.V(x)

      # perform linear operation and split into h heads

      k = k.view(-1, self.h, self.d_k)
      q = q.view(-1, self.h, self.d_k)
      v = v.view(-1, self.h, self.d_k)

      # transpose to get dimensions [n_nodes, attention_dim, n_heads]

      k = k.transpose(1, 2)
      q = q.transpose(1, 2)
      v = v.transpose(1, 2)

      src = q[edge[0, :], :, :]
      dst_k = k[edge[1, :], :, :]

    if not self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      prods = self.output_var ** 2 * torch.exp(-(torch.sum((src - dst_k) ** 2, dim=1) / (2 * self.lengthscale ** 2)))
    elif self.opt['attention_type'] == "scaled_dot":
      prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)
    elif self.opt['attention_type'] == "cosine_sim":
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
      prods = cos(src, dst_k)
    elif self.opt['attention_type'] == "pearson":
      src_mu = torch.mean(src, dim=1, keepdim=True)
      dst_mu = torch.mean(dst_k, dim=1, keepdim=True)
      src = src - src_mu
      dst_k = dst_k - dst_mu
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
      prods = cos(src, dst_k)

    if self.opt['reweight_attention'] and self.edge_weights is not None:
      prods = prods * self.edge_weights.unsqueeze(dim=1)
    if self.opt['square_plus']:
      attention = squareplus(prods, edge[self.opt['attention_norm_idx']])
    else:
      attention = softmax(prods, edge[self.opt['attention_norm_idx']])
    return attention, (v, prods)

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'