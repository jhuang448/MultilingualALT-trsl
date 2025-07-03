# Multilingual Joint Training

The following script trains a monolingual model:
```
python train.py hparams/1lang.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder
```

To change the number of language involved, switch to other yaml files under `hparams`.