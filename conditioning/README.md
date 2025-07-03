# Language Conditioning

## Language-Informed Models

To train a language-informed model with language embedding:
```
python train_emb.py hparams/emb_enc.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder
```

To train a language-informed model with FiLM conditioning layer:
```
python train_film.py hparams/film123.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder
```

To change the conditioning position, switch to other yaml files under `hparams`.

## Optionally Language-Informed Models

(i.e. the model can run without language information)


To train a language-informed model with unknown language embedding:
```
python train_emb_unk.py.py hparams/emb_both_unk.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder
```

To train a language-informed model with FiLM conditioning layer with a switch:
```
python train_film_onoff.py hparams/film123.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder
```

