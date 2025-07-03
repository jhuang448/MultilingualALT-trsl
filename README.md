# Multilingual Lyrics Transcription

This repository is for an extended work from the following paper: 

Jiawen Huang, Emmanouil Benetos, “**Towards Building an End-to-End Multilingual Automatic Lyrics Transcription Model**”, 
32th European Signal Processing Conference, Lyon, France, 2024. 
[Link](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/97337/Huang%20Towards%20Building%20an%202024%20Accepted.pdf?sequence=2&isAllowed=y)

The paper for this repository is under review. **trsl** stands for **transliteration**, rephrased as **phonetic reinterpretation** in the paper. You can also perceive it as **soramimi** (https://en.wikipedia.org/wiki/Soramimi).

![illustration](./fig/illustrations-fr.png)

## Dependencies

This repo is written in python 3.7 using the speechbrain toolkit v0.5.14.

## Preparing Data

### Datasets

The **DALI v2** and **MulJam v2** are used for training in this work. The **MultiLang Jamendo v1** is used for evaluation

See instructions on how to get the latest versions of the datasets: 

DALI v2: [https://github.com/gabolsgabs/DALI](https://github.com/gabolsgabs/DALI). 

MulJam v2: [https://github.com/jhuang448/ALT-Preprocess](https://github.com/jhuang448/ALT-Preprocess)

MultiLang Jamendo v1.1 (though we used v1.0, which is not longer available now): [https://huggingface.co/datasets/jamendolyrics/jamendolyrics](https://huggingface.co/datasets/jamendolyrics/jamendolyrics)

All songs are **source-separated** and **segmented into utterances**. The utterances are organized as the following structure:

```
$muljam_data_folder
├── train
├── valid
    ├── 1331475_9.wav
    ├── ...
$dali_data_folder
├── train
├── valid
    ├── 4058aa5a512d42cb931e37b87120dae5_0.wav
    ├── ...
$jamendo_data_folder
├── jamendolyrics
    ├── Ridgway_-_Fire_Inside_33.wav
    ├── ...
```
We provide a repository that handles the preprocessing for lyrics transcription: [https://github.com/jhuang448/ALT-Preprocess/](https://github.com/jhuang448/ALT-Preprocess/)


A few notes on version changes of the datasets: 
1. MulJam: In this project we were using the MulJam v2 version. MulJam v1 is slightly different in the train/valid split. 
MulJam v2 is recommended as it removes overlapping songs in MultiLang Jamendo.
2. Multi-Lang Jamendo: One French song from the MultiLang Jamendo v1 dataset has been removed from the official repository to form v1.1. Multi-Lang Jamendo v1.1 is recommended.

### Data splits

The training splits are constructed from the split here: [https://github.com/jhuang448/ALT-Preprocess/tree/main/input/preconstructed-split](https://github.com/jhuang448/ALT-Preprocess/tree/main/input/preconstructed-split). You may use the scripts to prepare the segmented and separated data.

For training, please download the data split [here](https://drive.google.com/drive/folders/1nPrTA0VMPkWQRow-Oj9o7h_6BPSca9ip?usp=sharing).
```
csv_folder
├── 1lang
├── 4lang
├── 5lang
├── 6lang
```

## Training

Instructions are provided in the corresponding folders: *joint-training* (multilingual joint training), *conditioning* (language conditioning), and *transliteration* (phonetic reinterpretation).

## Cite this work

Coming soon (after the extended work is published)!

Cite the EUSIPCO paper:

```
@inproceedings{DBLP:conf/eusipco/HuangB24,
  author       = {Jiawen Huang and
                  Emmanouil Benetos},
  title        = {Towards Building an End-to-End Multilingual Automatic Lyrics Transcription Model},
  booktitle    = {32nd European Signal Processing Conference, {EUSIPCO} 2024, Lyon,
                  France, August 26-30, 2024},
  pages        = {146--150},
  publisher    = {{IEEE}},
  year         = {2024},
}
```