import torch
import torch.nn as nn
import torch.nn.functional as F

class GBN(nn.Module):
    def __init__(self,inp,vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum = momentum)
        self.vbs = vbs
    def forward(self, x):
        chunks = max(1, x.size(0) // self.vbs)
        chunk = torch.chunk(x,chunks,0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res,0)
    
class AttentionTransformer(nn.Module):
    def __init__(self,d_a,inp_dim, relax, vbs=128):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.r = relax

    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = F.softmax(a * priors, dim=-1)
        priors = priors * (self.r - mask)
        return mask
    
class GLU(nn.Module):
    def __init__(self, inp_dim, out_dim, fc=None, vbs=128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim*2)
        self.bn = GBN(out_dim*2, vbs=vbs)
        self.od = out_dim
    def forward(self,x):
        x = self.bn(self.fc(x))
        return x[:,:self.od]*torch.sigmoid(x[:,self.od:])
    
class FeatureTransformer(nn.Module):
    def __init__(self,inp_dim, out_dim, shared, n_ind, vbs=128):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fx in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fx, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim,out_dim, vbs=vbs))
        self.register_buffer("scale", torch.sqrt(torch.tensor(0.5)))
    
    def forward(self,x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x*self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x*self.scale
        return x
        
class DecisionStep(nn.Module):
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, n_d+n_a, shared, n_ind, vbs)
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
    def forward(self,x,a,priors):
        mask = self.atten_tran(a,priors)
        sparse_loss = ((-1)*mask*torch.log(mask+1e-10)).mean()
        x = self.fea_tran(x*mask)
        return x, sparse_loss
    
class TabNet(nn.Module):
    def __init__(self, inp_dim, final_out_dim, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=128):
        super().__init__()
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2*(n_d+n_a)))
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(n_d+n_a, 2*(n_d+n_a)))
        else:
            self.shared=None
        self.first_step = FeatureTransformer(inp_dim, n_d+n_a, self.shared, n_ind)
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs))
        self.fc = nn.Linear(n_d, final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d
    def forward(self, x):
        x = self.bn(x)
        x_a = self.first_step(x)[:,self.n_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x,x_a,priors)
            out += F.relu(x_te[:,:self.n_d])
            x_a = x_te[:,self.n_d:]
            sparse_loss += 1
        return self.fc(out), sparse_loss
