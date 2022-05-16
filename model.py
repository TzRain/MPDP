import torch as t
from torch import nn
import numpy as np

class MPDP(nn.Module):
    def __init__(self, n_prot, n_drug, hidden_dim, seq_emb_dim = 1900):
        super(MPDP, self).__init__()

        self.n_prot = n_prot
        self.n_drug = n_drug
        self.hidden_dim = hidden_dim
        self.seq_emb_dim = seq_emb_dim
        #? Embedding layer load from static represatation data pemb:UniRep demb:?
        self.pemb = nn.Embedding(n_prot, seq_emb_dim)
        self.demb = nn.Embedding(n_drug, seq_emb_dim)
        #TODO add Adversarial Part
        self.plinear = nn.Linear(seq_emb_dim, hidden_dim)
        self.dlinear = nn.Linear(seq_emb_dim, hidden_dim)
        self.dti_clf = nn.Linear(hidden_dim, 1)
        self.ppi_clf = nn.Linear(hidden_dim, 1)
        self.ddi_clf = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_parameters()

    def init_parameters(self):
        y = 1.0/np.sqrt(self.seq_emb_dim)
        y2 = 1.0/np.sqrt(self.hidden_dim)
        self.dti_clf.weight.data.uniform_(-y2, y2)
        self.dti_clf.bias.data.fill_(0)
        self.ppi_clf.weight.data.uniform_(-y2, y2)
        self.ppi_clf.bias.data.fill_(0)
        self.plinear.weight.data.uniform_(-y, y)
        self.plinear.bias.data.fill_(0)
        self.dlinear.weight.data.uniform_(-y, y)
        self.dlinear.bias.data.fill_(0)
        self.pemb.weight.data.uniform_(-y, y)
        self.demb.weight.data.uniform_(-y, y)

    def forward(self, pindex_tensor, dindex_tensor, dti_pairs,ppi_pairs,ddi_pairs):
        pemb = self.pemb(pindex_tensor)
        demb = self.demb(dindex_tensor)

        phid = self.plinear(pemb)
        dhid = self.dlinear(demb)

        pghid = self.relu(phid)
        dghid = self.relu(dhid)

        dti_out = self.sigmoid(self.dti_clf(dghid[dti_pairs[:,0]] * pghid[dti_pairs[:, 1]]))
        ppi_out = self.sigmoid(self.ppi_clf(pghid[ppi_pairs[:,0]] * pghid[ppi_pairs[:, 1]]))
        ddi_out = self.sigmoid(self.ddi_clf(dghid[ddi_pairs[:,0]] * dghid[ddi_pairs[:, 1]]))
        return dti_out.squeeze(), ppi_out.squeeze(), ddi_out.squeeze()

    def infer(self, pindex_tensor, dindex_tensor, dti_pairs):
        pemb = self.pemb(pindex_tensor)
        demb = self.demb(dindex_tensor)

        phid = self.plinear(pemb)
        dhid = self.dlinear(demb)

        pghid = self.relu(phid)
        dghid = self.relu(dhid)

        dti_out = self.sigmoid(self.dti_clf(dghid[dti_pairs[:,0]] * pghid[dti_pairs[:, 1]]))
        return dti_out.squeeze() 