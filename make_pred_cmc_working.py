import os
import anndata as ad
from cellot.utils.helpers import load_config
from cellot.utils.loaders import load
from cellot.data.cell import read_list
from cellot.data.cell import AnnDataDataset
from torch.utils.data import DataLoader
from cellot.data.utils import cast_loader_to_iterator
from cellot.utils.helpers import nest_dict, flat_dict
from scipy import sparse
def predict_from_unstim_data(result_path, unstim_data_path, output_path,stim):
    config_path = os.path.join(result_path, "config.yaml")
    chkpt = os.path.join(result_path, "cache/model.pt")

    # load the config and then the model (f,g)
    config = load_config(config_path)
    (_, g), _, _ = load(config, restore=chkpt)
    g.eval()
    # load the data to predict and filter with the interzsting markers
    anndata_to_predict = ad.read(unstim_data_path)
    features2 = ['149Sm_pCREB','159Tb_pMAPKAPK2','166Er_pNFkB','151Eu_pp38', '155Gd_pS6','153Eu_pSTAT1', '154Sm_pSTAT3','174Yb_HLADR']
    features = ['149Sm_CREB','150Nd_STAT5','151Eu_p38','153Eu_STAT1','154Sm_STAT3','155Gd_S6', '159Tb_MAPKAPK2','164Dy_IkB', '166Er_NFkB','168Er_pSTAT6', '169Tm_CD25','174Yb_HLADR','167Er_ERK']
    features = ['149Sm_pCREB','167Er_pERK12','164Dy_IkB','159Tb_pMAPKAPK2','166Er_pNFkB','151Eu_pp38', '155Gd_pS6','153Eu_pSTAT1', '154Sm_pSTAT3','150Nd_pSTAT5', '168Yb_pSTAT6','174Yb_HLADR','169Tm_CD25']
    anndata_to_predict = anndata_to_predict[:, features].copy()
    unstim_anndata_to_predict=anndata_to_predict[anndata_to_predict.obs['drug']=='Unstim'].copy()
    

    # predict the data (first put it in the dataset format)
    dataset_args = {}
    dataset = AnnDataDataset(unstim_anndata_to_predict.copy(), **dataset_args)
    loader2 = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader2))
    outputs = g.transport(inputs.requires_grad_(True)).detach().numpy()
    predicted = ad.AnnData(
        outputs,
        obs=dataset.adata.obs.copy(),
        var=dataset.adata.var.copy(),
    )
    predicted.obs['drug']=stim
    original_anndata=anndata_to_predict.copy()
    original_anndata.obs["state"] = "true_corrected"
    
    predicted.obs["state"] = "predicted"
    concatenated = ad.concat([predicted, original_anndata], axis=0)
    print(concatenated.obs['drug'].unique(), 'concat')
    print(predicted.obs['drug'].unique(), 'predicted')
    # save the prediction in the desired format
    if output_path.endswith(".csv"):
        concatenated = concatenated.to_df()
        concatenated.to_csv(output_path)

    elif output_path.endswith(".h5ad"):
        print(output_path)
        concatenated.write(output_path)
    return

perio_stim_list_=['P._gingivalis']
perio_cell_list_=['Classical_Monocytes_(CD14+CD16-)']
for cell_type in perio_cell_list_:
    for stim in perio_stim_list_:
        unstim_data_path = f'./datasets/surge_dbl_corrected_test/surge_concatenated_LPS_Classical_Monocytes_(CD14+CD16-).h5ad'
        result_path = f"./results/cMCS_LPS_perio_dbl_corr/test_perio/model-cellot"
        output_path = f'./results/cMCS_LPS_perio_dbl_corr/test_perio/model-cellot/pred_surge_corrected_only_unstim.h5ad'
        ada = predict_from_unstim_data(result_path, unstim_data_path, output_path,'LPS')
        print(f"Predicted {cell_type} for {stim} perio")