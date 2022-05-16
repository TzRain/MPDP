# MPDP
MPDP：Multitask-Prediction-between-Drug-Protein

## how to run
python version 3.7.x
`pip install -r requirements.txt`
embedding data is already generated just 
`python mpdp.py`  


## Data
#### Embedding lookup table
- drug_features.csv size: `n*dim(1900)`
- protein_features.csv size:`n*dim(1900)`

#### DDT ground truth
- ddi_edgelist.csv : `idx idx`
- ddi_edgeweight.csv : `inter-scorce(float)`

#### PPT ground truth
- ppi_edgelist.csv : `idx idx`
- ppi_edgeweight.csv : `inter-scorce(float)`

#### DTI data
- neg_test_idx.csv : `idx idx`
- neg_train_idx.csv : `idx idx`
- pos_test_idx.csv : `idx idx`
- pos_train_idx.csv : `idx idx`
> to-do list as below
### Data Step
1. get protein and drug categories
2. gen they embedding
3. gen DTI DDI PPI pairs

## Building-Step
1.Read org MTT

2.Modify MTT to multitask PPI DTI PPI
- how to design new loss_fn  :(
    - $loss  = l_{dti} + w_{ppi} * l_{ppi} + w_{ddi} * l_{ddi}$
    - which task is main task ?
- data for PPI DDI need spite into test\train\val?

3.add adversaial task 