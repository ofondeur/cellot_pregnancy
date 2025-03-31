import os

# Définition des listes de stimulations et de types cellulaires
perio_stim_list = ['TNFa', 'P. gingivalis', 'IFNa']
perio_cell_list = [
    'Granulocytes (CD45-CD66+)', 'B-Cells (CD19+CD3-)', 'Classical Monocytes (CD14+CD16-)',
    'MDSCs (lin-CD11b-CD14+HLADRlo)', 'mDCs (CD11c+HLADR+)', 'pDCs(CD123+HLADR+)',
    'Intermediate Monocytes (CD14+CD16+)', 'Non-classical Monocytes (CD14-CD16+)',
    'CD56+CD16- NK Cells', 'CD56loCD16+NK Cells', 'NK Cells (CD7+)',
    'CD4 T-Cells', 'Tregs (CD25+FoxP3+)', 'CD8 T-Cells', 'CD8-CD4- T-Cells'
]

# Création du dossier de sortie
config_dir = "./yaml/sherlock_perio"
os.makedirs(config_dir, exist_ok=True)

# Modèle de configuration YAML
template_yaml = """data:
  features: ./datasets/sherlock_training_data/features.txt
  path: {data_path}
  condition: drug
  source: Unstim
  type: cell

dataloader:
  batch_size: 256
  shuffle: true

datasplit:
  groupby: drug
  name: train_test
  test_size: 0.2
"""

# Générer les fichiers de configuration
for stim in perio_stim_list:
    for cell in perio_cell_list:
        data_path = f"./datasets/sherlock_training_data/combined_{stim}_{cell.replace(' ', '_')}_train.h5ad"
        config_content = template_yaml.format(data_path=data_path)

        config_filename = f"sherlock_{stim}_{cell.replace(' ', '_')}.yaml"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            f.write(config_content)

        print(f"Config generated: {config_filepath}")
