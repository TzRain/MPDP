import pandas as pd

from model import *
from sklearn.model_selection import train_test_split
import argparse
from utils.utils import *

def read_int_data(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    int_lbl = [1] * len(pos_df) + [0] * len(neg_df)
    return int_edges, t.LongTensor(int_edges), t.FloatTensor(int_lbl)

def read_train_data(pos_path, neg_path, fixval=False):
    pos_df = pd.read_csv(pos_path).values.tolist()
    pos_train, pos_val = train_test_split(pos_df, test_size=0.1, random_state=42)

    neg_df = pd.read_csv(neg_path).values.tolist()
    indexes = list(range(len(neg_df)))
    random.shuffle(indexes)
    selected_indexes = indexes[:int(10*len(pos_df))] #* neg_num <= pos_num * 10
    neg_df = [neg_df[item] for item in selected_indexes]

    neg_train, neg_val = train_test_split(neg_df, test_size=0.1, random_state=42)
    if not fixval:
        train_data = pos_train + neg_train
        train_lbl = [1] * len(pos_train) + [0] * len(neg_train)

        val_data = pos_val + neg_val
        val_lbl = [1] * len(pos_val) + [0] * len(neg_val)
        val_tensor = t.LongTensor(val_data)
        val_lbl_tensor = t.FloatTensor(val_lbl)
    else:
        train_data = pos_train + neg_train
        train_lbl = [1] * len(pos_train) + [0] * len(neg_train)
        val_data, val_tensor, val_lbl_tensor = read_int_data(pos_path.replace('train', 'val'), neg_path.replace('train', 'val'))

    return train_data, val_data, t.LongTensor(train_data), t.FloatTensor(train_lbl), val_tensor, val_lbl_tensor

def save_model(model, save_path):
    t.save(model.state_dict(), save_path)

def load_model(model, model_path):
    model.load_state_dict(t.load(model_path))

def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_runs', type=int, default=10, metavar='N',help='number of experiment runs')
    parser.add_argument('--data_dir', default='db/fake/', help='dataset directory')
    parser.add_argument('--protein_feature_path', default='protein_features.csv', help='protein_feature path')
    parser.add_argument('--drug_feature_path', default='drug_features.csv', help = 'drug_feature path')
    parser.add_argument('--ppi_edge_list', default='ppi_edgelist.csv', help='ppi edge list path')
    parser.add_argument('--ppi_edge_weight', default='ppi_edgeweight.csv', help='ppi edge weight path')
    parser.add_argument('--ddi_edge_list', default='ddi_edgelist.csv', help='ddi edge list path')
    parser.add_argument('--ddi_edge_weight', default='ddi_edgeweight.csv', help='ddi edge weight path')
    parser.add_argument('--pos_train_path', default='pos_train_idx.csv', help='pos train path')
    parser.add_argument('--pos_test_path', default='pos_test_idx.csv', help='pos test path')
    parser.add_argument('--neg_train_path', default='neg_train_idx.csv', help='neg train path')
    parser.add_argument('--neg_test_path', default='neg_test_idx.csv', help='neg test path')
    parser.add_argument('--fixval', default=False, help='use the fix validation set or not')
    
    args = parser.parse_args()

    args.data_dir = standardize_dir(args.data_dir)
    negname = 'mtt_' + args.neg_test_path.replace('.csv', '')
    args.protein_feature_path = args.data_dir + args.protein_feature_path
    args.drug_feature_path = args.data_dir + args.drug_feature_path
    args.ppi_edge_list = args.data_dir + args.ppi_edge_list
    args.ppi_edge_weight = args.data_dir + args.ppi_edge_weight
    args.ddi_edge_list = args.data_dir + args.ddi_edge_list
    args.ddi_edge_weight = args.data_dir + args.ddi_edge_weight
    args.pos_train_path = args.data_dir + args.pos_train_path
    args.pos_test_path = args.data_dir + args.pos_test_path
    args.neg_train_path = args.data_dir + args.neg_train_path
    args.neg_test_path = args.data_dir + args.neg_test_path
    args.n_runs = int(args.n_runs)
    args.fixval = bool(args.fixval)
    
    
    drug_features = t.FloatTensor(pd.read_csv(args.drug_feature_path, header=None).values)
    protein_features = t.FloatTensor(pd.read_csv(args.protein_feature_path, header=None).values)
    n_protein = protein_features.size(0)
    n_drug = drug_features.size(0)
    print('Finish loading features')
    
    ddi_edgeweight = t.FloatTensor(pd.read_csv(args.ddi_edge_weight).values)
    ddi_edgelist = t.LongTensor(pd.read_csv(args.ddi_edge_list).values)
    print('Finish loading DDI')
    
    ppi_edgeweight = t.FloatTensor(pd.read_csv(args.ppi_edge_weight).values)
    ppi_edgelist = t.LongTensor(pd.read_csv(args.ppi_edge_list).values)
    print('Finish loading PPI')
    
    pindex_tensor = t.LongTensor(list(range(n_protein)))
    dindex_tensor = t.LongTensor(list(range(n_drug)))
    
    pos_train_pairs, val_data, train_tensor, train_lbl_tensor, val_tensor, val_lbl_tensor = read_train_data(args.pos_train_path, args.neg_train_path, args.fixval)
    test_pairs, test_tensor, test_lbl_tensor = read_int_data(args.pos_test_path, args.neg_test_path)
    test_lbl = test_lbl_tensor.detach().numpy()
    val_lbl = val_lbl_tensor.detach().numpy()
    print('Finish loading int pairs')
    
    ppi_edgeweight = ppi_edgeweight.view(-1)
    ddi_edgeweight = ddi_edgeweight.view(-1)
    
    criterion = t.nn.BCELoss()
    criterion2 = t.nn.MSELoss()
    
    if t.cuda.is_available():
        ppi_edgelist = ppi_edgelist.cuda()
        ppi_edgeweight = ppi_edgeweight.cuda()
        ddi_edgelist = ddi_edgelist.cuda()
        ddi_edgeweight = ddi_edgeweight.cuda()
        pindex_tensor = pindex_tensor.cuda()
        dindex_tensor = dindex_tensor.cuda()
        protein_features = protein_features.cuda()
        drug_features = drug_features.cuda()
        train_tensor = train_tensor.cuda()
        test_tensor = test_tensor.cuda()
        train_lbl_tensor = train_lbl_tensor.cuda()
        criterion = criterion.cuda()
        criterion2 = criterion2.cuda()
        val_tensor = val_tensor.cuda()
    
    #? tuning multi-task wich is main-task
    max_auc = [0,0]
    lrs = [0.001, 0.01]
    grid_epochs = [200]
    hiddens = [8,16,32,64]
    ppi_weights = [1e-4, 1e-3, 1e-2, 1e-1, 1] 
    ddi_weights = [1e-4, 1e-3, 1e-2, 1e-1, 1]  
    adv_weights = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    diff_weights = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    model_prefix = args.data_dir + negname + '_'
    performance_dict = dict()
    val_performance_dict = dict()
    
    def score_loss(dti,ppi,ddi,args):
        return criterion(dti, train_lbl_tensor) + args.ppi_weight * criterion2(ppi, ppi_edgeweight) + args.ddi_weight * criterion2(ddi, ddi_edgeweight)
    
    def adv_loss(dti,ppi,ddi,args):
        return 0
        
    def diff_loss(pghid_ppi,pghid_dti,dghid_ddi,dghid_dti,args):
        p_ppi = pghid_ppi[:args.hid]
        p_dti = pghid_dti[:args.hid]
        p_share = pghid_dti[args.hid:]
        d_ddi = dghid_ddi[:args.hid]
        d_dti = dghid_dti[:args.hid]
        d_share = dghid_dti[args.hid:]
        return 0
    
    
    for lr in lrs:
        for ppi_weight in ppi_weights:
            args.ppi_weight = ppi_weight
            for ddi_weight in ddi_weights:
                args.ddi_weight = ddi_weight
                for adv_weight in adv_weights:
                    args.adv_weight = adv_weight
                    for diff_weight in diff_weights:
                        args.adv_weight = diff_weight
                        for hid in hiddens:
                            for epochs in grid_epochs:
                                
                                # training setting 
                                args.epochs = epochs
                                params = [epochs, ppi_weight, ddi_weight,adv_weight,diff_weight, hid, lr, 0]
                                params = [str(item) for item in params]
                                save_model_prefix = model_prefix + '_'.join(params)  + '.model'
                                
                                all_dtis = list()
                                val_all_dtis = list()
                                
                                # start training 
                                for irun in range(args.n_runs):
                                    save_model_path = save_model_prefix.replace('0.model', str(irun) + '.model')
                                    model = AdvMPDP(n_protein, n_drug, hid)
                                    model.pemb.weight.data = protein_features
                                    model.demb.weight.data = drug_features
                                    optimizer = t.optim.Adam(model.parameters(), lr=lr)
                                    if t.cuda.is_available():
                                        model = model.cuda()
                                    best_ap = 0
                                    for epoch in range(0, epochs):
                                        model.train()
                                        optimizer.zero_grad()
                                        dti_out, ppi_out, ddi_out,dti_out_share, ppi_out_share, ddi_out_share,pghid_ppi,pghid_dti,dghid_ddi,dghid_dti = model(pindex_tensor, dindex_tensor,train_tensor, ppi_edgelist,ddi_edgelist)
                                        loss = score_loss(dti_out, ppi_out, ddi_out,args) + adv_weight * adv_loss(dti_out_share, ppi_out_share, ddi_out_share,args) + diff_weight * diff_loss(pghid_ppi,pghid_dti,dghid_ddi,dghid_dti,args)
                                        loss.backward()
                                        optimizer.step()
                                        loss_val = loss.item() if not t.cuda.is_available() else loss.cpu().item()
                                        #? why / train_lbl_tensor.size(0)
                                        print('Epoch: ', epoch, ' loss: ', loss_val / train_lbl_tensor.size(0))
                                        
                                        if epoch % 2 == 0:
                                            model.eval()
                                            pred_dti = model.infer(pindex_tensor, dindex_tensor, val_tensor)
                                            pred_dti = pred_dti.detach().numpy() if not t.cuda.is_available() else pred_dti.cpu().detach().numpy()
                                            val_pred_lbl = pred_dti.tolist()
                                            val_pred_lbl = [item[0] if type(item) == list else item for item in val_pred_lbl]
                                            auc_dti, aupr_dti, sn, sp, acc, topk, precision, recall, f1 = get_score(val_lbl, val_pred_lbl)
                                            print('Validation set lr:%.4f, auc:%.4f, aupr:%.4f' %(lr, auc_dti, aupr_dti))
                                            if best_ap < aupr_dti:
                                                best_ap = aupr_dti
                                                save_model(model, save_model_path) # Best model on the validation set
                                    # show best model 
                                    print('_'.join(params), 'best_ap: ', best_ap)
                                    model = AdvMPDP(n_protein, n_drug, hid)
                                    if t.cuda.is_available():
                                        model = model.cuda()
                                    load_model(model, save_model_path)
                                    os.remove(save_model_path)
                                    model.eval()
                                    # performance on the testing set
                                    pred_dti = model.infer(pindex_tensor, dindex_tensor,test_tensor)
                                    pred_dti = pred_dti.detach().numpy() if not t.cuda.is_available() else pred_dti.cpu().detach().numpy()
                                    test_pred_lbl = pred_dti.tolist()
                                    test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]
                                    
                                    topks = list()
                                    for k in range(1,11):
                                        auc_dti, aupr_dti, sn, sp, acc, topk, precision, recall, f1 = get_score(test_lbl, test_pred_lbl,K=k)
                                        topks.append(topk)
                                    all_dtis.append([auc_dti, aupr_dti, sn, sp, acc, topks[0], precision, recall, f1] + topks)
                                    print(params, all_dtis[-1])
                                    
                                    # Save the performance on the validation set also
                                    pred_dti = model.infer(pindex_tensor, dindex_tensor,val_tensor)
                                    pred_dti = pred_dti.detach().numpy() if not t.cuda.is_available() else pred_dti.cpu().detach().numpy()
                                    
                                    test_pred_lbl = pred_dti.tolist()
                                    test_pred_lbl = [item[0] if type(item) == list else item for item in test_pred_lbl]
                                    auc_dti, aupr_dti, sn, sp, acc, topk, precision, recall, f1 = get_score(val_lbl, test_pred_lbl)
                                    val_all_dtis.append([auc_dti, aupr_dti, sn, sp, acc, topk, precision, recall, f1])#, sn, sp, acc, topk])
                                    if max_auc[0] < auc_dti:
                                        max_auc = [auc_dti, aupr_dti]
                                        
                                t.cuda.empty_cache()
                                arr = np.array(all_dtis)
                                print('all_dtis: ', all_dtis)
                                mean = np.mean(arr, axis=0)
                                std = np.std(arr, axis=0)
                                print('Mean auc_dti, aupr_dti, sn, sp, acc, topk:')
                                print(mean)
                                print('Std auc_dti, aupr_dti, sn, sp, acc, topk:')
                                print(std)
                                print('max auc, aupr:', max_auc)

                                performance_dict['_'.join(params[:7])] = [all_dtis, list(mean), list(std)]
                                arr = np.array(val_all_dtis)
                                print('all_dtis: ', all_dtis)
                                mean = np.mean(arr, axis=0)
                                std = np.std(arr, axis=0)
                                val_performance_dict['_'.join(params[:7])] = [val_all_dtis, list(mean), list(std)]
                        
    writer = open(args.data_dir + negname + 'grid_search_res.txt', 'w')
    print('write to file:', args.data_dir + negname + 'grid_search_res.txt')
    maxf1 = 0
    max_key = 0
    for key in performance_dict:
        writer.write('Result for ' + str(key) + '\n')
        writer.write(str(performance_dict[key][0]) + '\n')
        writer.write('mean: ' + str(performance_dict[key][1]) + '\n')
        writer.write('std: ' + str(performance_dict[key][2]) + '\n')

        print('Result for ' + str(key) + '\n')
        print(str(performance_dict[key][0]) + '\n')
        print('mean: ' + str(performance_dict[key][1]) + '\n')
        print('std: ' + str(performance_dict[key][2]) + '\n')

        if val_performance_dict[key][1][-1] > maxf1:
            maxf1 = val_performance_dict[key][1][-1]
            max_key = key

    writer.write('Best results: ' + str(maxf1) + ' key: ' + str(max_key) + '\n')
    print('Best results: ' + str(performance_dict[max_key]) + ' key: ' + str(max_key) + '\n')
    writer.close()

if __name__ == "__main__":
    main()
