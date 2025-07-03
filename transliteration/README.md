# Transliteration / Phonetic Reinterpretation

This folder generates phonetic reinterpretation, run inference with the models trained in previous steps.

First, for convenience, create a softlink to the results folder:

```
ln -s ../conditioning/results/ results
```

Download language models for inference [here](https://drive.google.com/drive/folders/1nPrTA0VMPkWQRow-Oj9o7h_6BPSca9ip?usp=sharing).

## Generate phonetic reinterpretation

The following script generates phonetic reinterpretation to target language (**en**) on a french set (**train-fr.csv**). The model used is specified by **exp_name** and **attempt**, which should be identical to the settings used in a film conditioning training in the last step.
```
python tran-inf.py hparams/baseline_LMinf.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder --exp_name film_onoff --attempt 123 --target_lang en --test_csv [./csv_folder/1lang/train-fr.csv]
```
## Combine the ground truth and the phonetic reinterpretation

The generated csv file is saved under the same results folder, and can be later combined with ground truth for data augmentation experiments using **combined.ipynb**.

## Train with phonetic reinterpretation

The following script trains a film123 model with phonetic reinterpretation, at an augmentation rate of 0.3. The **train_csv** is obtained from the notebook from last step.
```
python train_tran.py hparams/film123.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder --train_csv ./csv_folder/tran/film123/tran-4lang.csv --aug_rate 0.3 --attempt film123-0.3-4lang
```

Similarly, to train a film123-switch model, use train_tran_switch.py instead:
```
python train_tran_switch.py hparams/film123.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder  --train_csv ./csv_folder/tran/film123/tran-4lang.csv --aug_rate 0.3 --attempt film123_onoff-0.3-4lang
```

## Inference with Conditioning and LM
```
python tran-inf-wer.py hparams/baseline_LMinf.yaml --muljam_data_folder $muljam_data_folder --dali_data_folder $dali_data_folder --jamendo_data_folder $jamendo_data_folder --exp_name tran --attempt film123-0.3-4lang --test_beam_size 66 --lm_weight 0.1
```