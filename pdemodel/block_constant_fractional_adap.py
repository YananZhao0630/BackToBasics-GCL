from pdemodel.base_classes import ODEblock
import torch
from pdemodel.utils import get_rw_adj, gcn_norm_fill_val
from torchfde import fdeint

class ConstantODEblock_FRAC_adap(ODEblock):
  def __init__(self, odefunc, opt, data, device, t=torch.tensor([0, 1])):
    super(ConstantODEblock_FRAC_adap, self).__init__(odefunc, opt, data, device, t)

    self.odefunc = odefunc(opt['hidden_dim'], opt['hidden_dim'], opt, data, device)
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
    # self.reg_odefunc = None
    # self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    # self.set_tol()
    self.device = device
    self.opt = opt
    self.data = data
    self.x_mean = data.x.mean(dim=0)


  def forward(self, x, adj):

    # t = self.t.type_as(x)

    # reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )

    # func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    # state = (x,) + reg_states if self.training and self.nreg > 0 else x
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
                                           num_nodes=x.size(0),
                                           dtype=x.dtype)
      # edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
      #                                    fill_value=self.opt['self_loop_weight'],
      #                                    num_nodes=x.size(0),
      #                                    dtype=x.dtype)

    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)


    # integrator = self.train_integrator if self.training else self.test_integrator

    func = self.odefunc
    state = x


    # alpha = torch.tensor(self.opt['alpha_ode'])
    # 1) figure out which device alpha_net lives on
    alpha_module = self.opt['alpha_net']
    # alpha_input = self.data.x
    alpha = alpha_module(self.x_mean)  # → tensor on cuda:3

    # alpha = self.opt['alpha_net'](self.data).to(x.device)

    if alpha > 1:
        raise ValueError("alpha_ode must be in (0,1)")

    z = fdeint(func, state, alpha, t=self.opt['time'], step_size=self.opt['step_size'], method=self.opt['method'])
    print(alpha)
    return z, alpha

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
