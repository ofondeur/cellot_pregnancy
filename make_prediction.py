import os
import anndata as ad
from cellot.utils.helpers import load_config
from cellot.utils.loaders import load
from cellot.data.cell import read_list
from cellot.data.cell import AnnDataDataset
from torch.utils.data import DataLoader


def predict_from_unstim_data(
    result_path, unstim_data_path, pred_format, output_path=None
):
    config_path = os.path.join(result_path, "config.yaml")
    chkpt = os.path.join(result_path, "cache/model.pt")

    # load the config and then the model (f,g)
    config = load_config(config_path)
    (_, g), _, _ = load(config, restore=chkpt)
    g.eval()

    # load the data to predict and filter with the interzsting markers
    unstim_anndata_to_predict = ad.read(unstim_data_path)

    # features = read_list(config.data.features)
    features = [
        "151Eu_pP38",
        "149Sm_pCREB",
        "159Tb_pMK2",
        "167Er_pERK",
        "155Gd_pS6",
        "166Er_pNFkB",
        "150Nd_pSTAT5",
        "153Eu_pSTAT1",
        "154Sm_pSTAT3",
        "164Er_IkB",
        "168Er_pSTAT6",
    ]

    unstim_anndata_to_predict = unstim_anndata_to_predict[:, features].copy()
    unstim_anndata_to_predict = unstim_anndata_to_predict[
        unstim_anndata_to_predict.obs["condition"] == "control"
    ]

    # predict the data (first put it in the dataset format)
    dataset_args = {}
    dataset = AnnDataDataset(unstim_anndata_to_predict.copy(), **dataset_args)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))

    outputs = g.transport(inputs.requires_grad_(True)).detach().numpy()
    predicted = ad.AnnData(
        outputs,
        obs=dataset.adata.obs.copy(),
        var=dataset.adata.var.copy(),
    )

    # save the prediction in the desired format
    if pred_format == "csv":
        predicted = predicted.to_df()
        if output_path is not None:
            predicted.to_csv(output_path)
            return

    elif pred_format == "h5ad" and output_path is not None:
        predicted.write(output_path)
        return
    return predicted


# tests
result_path = "results/PTB/LPS_cMC/model-cellot"
unstim_data_path = "datasets/PTB_training/LPS_cMC_HV01.h5ad"
output_path = "results/PTB/LPS_cMC/model-cellot/pred.csv"
ada = predict_from_unstim_data(result_path, unstim_data_path, "csv", output_path)
