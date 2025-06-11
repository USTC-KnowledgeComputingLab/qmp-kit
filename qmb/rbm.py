
import torch
from  .mlp import MLP
from .bitspack import pack_int, unpack_int

class RBM(torch.nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super().__init__()
        self.visible_dim=visible_dim
        self.hidden_dim=hidden_dim
        self.W = torch.nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.01)  # ??
        self.visible_bias = torch.nn.Parameter(torch.zeros((visible_dim,1)))  # ??
        self.hidden_bias = torch.nn.Parameter(torch.zeros((hidden_dim,1)))  # ??
    
    def probability(self, v: torch.Tensor):
        batch_size=v.size()[0]
        e1=(v@self.visible_bias).squeeze(1)
        mid2=v@self.W+self.hidden_bias.expand([batch_size, -1])
        e2=mid2.exp().add(1).log().sum(dim=1)
        p=(e1+e2).exp()
        return p
    
    def sample(self, n, init: torch.Tensor):
        conf=init.clone().view(1,-1)
        sampled_conf=[init]
        for i in range(n-1):  # init is also sampled thus one place is taken.
            # samp h
            midh=conf@self.W+self.hidden_bias.T
            ph=torch.sigmoid(midh)
            h=torch.bernoulli(ph)
            # samp v
            midv=h@self.W.T+self.visible_bias.T
            pv=torch.sigmoid(midv)
            conf=torch.bernoulli(pv)
            sampled_conf.append(conf)
        conf_tensor=torch.cat(sampled_conf, dim=0)
        return conf_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probability(x)

class WaveFunctionNormal(torch.nn.Module):
    def __init__(
            self, 
            *, 
            sites: int, 
            physical_dim: int, 
            is_complex: bool, 
            rbm_hidden_dim: int,
            mpl_hidden_size: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.sites: int = sites
        # ?? assertion?
        self.rbm_hidden_dim: int = rbm_hidden_dim
        self.mpl_hidden_size: tuple[int, ...] = mpl_hidden_size

        # Build Networks
        self.probability: RBM = RBM(self.sites, self.rbm_hidden_dim)
        self.phase: torch.nn.Module = MLP(self.sites, 1, self.mpl_hidden_size)

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        batch_size: int = x.shape[0]
        x = unpack_int(x, size=1, last_dim=self.sites).view([batch_size, self.sites])
        x_float: torch.Tensor = x.to(dtype=dtype)
        magnitude: torch.Tensor = self.probability(x_float).double()**0.5
        phase: torch.Tensor = self.phase(x_float).view([batch_size]).double()
        return magnitude*(phase*1j).exp()
    
    def generate_conf(self, batch_size):
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        samples=self.probability.sample(batch_size, torch.bernoulli(torch.ones(batch_size, device=device)*0.5))
        amplitude=self(samples)
        
        return samples, amplitude









