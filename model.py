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
    
    
class AdvMPDP(nn.Module):
    def __init__(self, n_prot, n_drug, hidden_dim, seq_emb_dim = 1900):
        super(AdvMPDP, self).__init__()

        self.n_prot = n_prot
        self.n_drug = n_drug
        self.hidden_dim = hidden_dim
        self.seq_emb_dim = seq_emb_dim
        #? Embedding layer load from static represatation data pemb:UniRep demb:?
        self.pemb = nn.Embedding(n_prot, seq_emb_dim)
        self.demb = nn.Embedding(n_drug, seq_emb_dim)
        #TODO add Adversarial Part
        self.plinear_share = nn.Linear(seq_emb_dim, hidden_dim)
        self.plinear_ppi = nn.Linear(seq_emb_dim, hidden_dim)
        self.plinear_dti = nn.Linear(seq_emb_dim, hidden_dim)
        self.dlinear_share = nn.Linear(seq_emb_dim, hidden_dim)
        self.dlinear_ddi = nn.Linear(seq_emb_dim, hidden_dim)
        self.dlinear_dti = nn.Linear(seq_emb_dim, hidden_dim)
        self.dti_clf = nn.Linear(hidden_dim, 1)
        self.ppi_clf = nn.Linear(hidden_dim, 1)
        self.ddi_clf = nn.Linear(hidden_dim, 1)
        self.dti_clf_share = nn.Linear(hidden_dim, 1)
        self.ppi_clf_share = nn.Linear(hidden_dim, 1)
        self.ddi_clf_share = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_parameters()

    def init_parameters(self):
        y = 1.0/np.sqrt(self.seq_emb_dim)
        y2 = 1.0/np.sqrt(self.hidden_dim)
        
        self.dti_clf_share.weight.data.uniform_(-y2, y2)
        self.dti_clf_share.bias.data.fill_(0)
        self.ppi_clf_share.weight.data.uniform_(-y2, y2)
        self.ppi_clf_share.bias.data.fill_(0)
        
        self.dti_clf.weight.data.uniform_(-y2, y2)
        self.dti_clf.bias.data.fill_(0)
        self.ppi_clf.weight.data.uniform_(-y2, y2)
        self.ppi_clf.bias.data.fill_(0)
        
        self.plinear_ppi.weight.data.uniform_(-y, y)
        self.plinear_ppi.bias.data.fill_(0)
        self.dlinear_ddi.weight.data.uniform_(-y, y)
        self.dlinear_ddi.bias.data.fill_(0)
        self.plinear_dti.weight.data.uniform_(-y, y)
        self.plinear_dti.bias.data.fill_(0)
        self.dlinear_dti.weight.data.uniform_(-y, y)
        self.dlinear_dti.bias.data.fill_(0)
        self.plinear_share.weight.data.uniform_(-y, y)
        self.plinear_share.bias.data.fill_(0)
        self.dlinear_share.weight.data.uniform_(-y, y)
        self.dlinear_share.bias.data.fill_(0)
        self.pemb.weight.data.uniform_(-y, y)
        self.demb.weight.data.uniform_(-y, y)

    def forward(self, pindex_tensor, dindex_tensor, dti_pairs,ppi_pairs,ddi_pairs):
        pemb = self.pemb(pindex_tensor)
        demb = self.demb(dindex_tensor)

        phid_ppi = self.plinear_ppi(pemb)
        dhid_ddi = self.dlinear_ddi(demb)
        phid_dti = self.plinear_dti(pemb)
        dhid_dti = self.dlinear_dti(demb)
        phid_share = self.plinear_share(pemb)
        dhid_share = self.dlinear_share(demb)

        phid_ppi = t.concat((phid_ppi,phid_share))
        phid_dti = t.concat((phid_dti,phid_share))
        
        dhid_ddi = t.concat((dhid_ddi,dhid_share))
        dhid_dti = t.concat((dhid_dti,dhid_share))

        pghid_ppi = self.relu(pghid_ppi)
        pghid_dti = self.relu(pghid_dti)
        dghid_ddi = self.relu(dghid_ddi)
        dghid_dti = self.relu(dghid_dti)
        
        pghid_share = self.relu(pghid_share)
        dghid_share = self.relu(dghid_share)

        dti_out = self.sigmoid(self.dti_clf(dghid_dti[dti_pairs[:,0]] * pghid_dti[dti_pairs[:, 1]]))
        ppi_out = self.sigmoid(self.ppi_clf(pghid_ppi[ppi_pairs[:,0]] * pghid_ppi[ppi_pairs[:, 1]]))
        ddi_out = self.sigmoid(self.ddi_clf(dghid_ddi[ddi_pairs[:,0]] * dghid_ddi[ddi_pairs[:, 1]]))
        
        dti_out_share = self.sigmoid(self.dti_clf_share(pghid_share[dti_pairs[:,0]] * pghid_share[dti_pairs[:, 1]]))
        ppi_out_share = self.sigmoid(self.ppi_clf_share(pghid_share[ppi_pairs[:,0]] * pghid_share[ppi_pairs[:, 1]]))
        ddi_out_share = self.sigmoid(self.ddi_clf_share(dghid_share[ddi_pairs[:,0]] * dghid_share[ddi_pairs[:, 1]]))
        
        return dti_out.squeeze(), ppi_out.squeeze(), ddi_out.squeeze(),dti_out_share.squeeze(),ppi_out_share.squeeze(),ddi_out_share.squeeze() ,pghid_ppi,pghid_dti,dghid_ddi,dghid_dti

    def infer(self, pindex_tensor, dindex_tensor, dti_pairs):
        pemb = self.pemb(pindex_tensor)
        demb = self.demb(dindex_tensor)

        phid_dti = self.plinear_dti(pemb)
        dhid_dti = self.dlinear_dti(demb)
        phid_share = self.plinear_share(pemb)
        dhid_share = self.dlinear_share(demb)

        phid_dti = t.concat((phid_dti,phid_share))
        dhid_dti = t.concat((dhid_dti,dhid_share))

        pghid_dti = self.relu(pghid_dti)
        dghid_dti = self.relu(dghid_dti)

        dti_out = self.sigmoid(self.dti_clf(dghid_dti[dti_pairs[:,0]] * pghid_dti[dti_pairs[:, 1]]))
        return dti_out.squeeze()