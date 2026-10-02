# TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG

<div align="center">

> 🚧 &nbsp;**Repository Under Construction**&nbsp; 🚧
>
> Code is actively being cleaned up and documented.
> Follow the [`hele/dev`](../../tree/hele/dev) branch for the latest work in progress.

</div>
This guide outlines the steps to perform multimodal depression detection using EEG and speech data. Follow these instructions in a step-by-step manner.

First of all, please make sure you gain acess to the dataset (http://MODMA.lzu.edu.cn) and put the dataset on the directory:

```bash
dataset/
|----- audio_lanzhou_2015-2
|----- EEG_128channels_resting_lanzhou_2015
```

## Create virtual environment
First, set up a Python virtual environment to manage dependencies.
```bash
python3 -m venv .
```

## Activate virtual environment
```bash
source bin/activate
```
After activating the environment, run this command to install all necessary packages

```bash
python3 -m pip install e .
```

## Preprocessing
The preprocessing stage involves creating a consistent dataset and preparing the data for feature extraction.

First of all, let's make a folder containing aligned subject across all modalities (EEG/Speech).

run 
```bash
python3 preprocessing/create_dataset.py
```
This script finds subjects who have both EEG and audio recordings and creates a unified dataset. The output will confirm the number of aligned subjects and generate the split_dataset folder.

The output should show this:

- EEG subjects: 53
- Audio subjects: 52
- ✅ Aligned subjects: 38


### Audio Transcription
For transcribing audio, you can run 
```bash
python3 preprocessing/audio_transcription.py
```
This command transcribes all audio files (*.wav) for each subject and saves the transcription as a CSV file within their respective subject folders.

### Speech Preprocessing
For speech proecessing, you can run 

```bash
python3 preprocessing/speech_preprocessing.py
```
This script filters, resamples, segments (into 5-second clips), and normalizes the audio files.

### EEG Preprocessing
For EEG Preprocessing, you can run

```bash
python3 preprocessing/eeg_preprocessing.py
```

This command filters the EEG data and segments it into 10-second clips.

## Features Extraction
This section describes how to extract features from each modality using various models and techniques.

### EEG Features
**a. CBRAMOD**
For extracting CBRAMOD embeddings, we can run 

```bash
python3 extract_features_eeg/extract_cbramod.py --PRETRAINED_WEIGHTS cbramod_pretrained_weights/pretrained-weights.pth
```

Apart from using original pretrained weights, we also used pretrained weights trained from Mumtaz which we put it under name pretrained_weights2.pth, you can just put/moidfy the pretrained weights path. For details how to yield the pretrained_weights2, we follow the finetuning process on CBRAMOD Github page. Please refer to the original CBRAMOD GitHub website (https://github.com/wjq-learning/CBraMod)/

```bash
python3 extract_features_eeg/extract_cbramod.py --PRETRAINED_WEIGHTS cbramod_pretrained_weights/pretrained-weights2.pth
```

**b. LaBraM**
```bash
python3 -m extract_features_eeg.extract_labram
```

**c. Handcrafted features**
```bash
python3 -m extract_features_eeg.extract_handcrafted_features
```

### Text Features
you can run:

```bash
python3 -m extract_features_text.extract_features_text
```
this will produce the embeddings from all models: XLNet, MpNet, Chinese BERT Base and Chinese Macbert. All will be saved under *split_dataset/[subject_id]/text_embeddings_[model_name].npy*

### Speech Features
For speech features from pre-trained models and mfccs features, you can run:
```bash
python3 -m extract_features_speech.extract_features_speech --feature_extractor [feature extractor] --encoder [encoder] --device [cpu / gpu]
```
Feature extractor options:
- chinese_hubert
- mfccs 
- slr53

Encoder options:
- cnn_bigru
- cnn_lstm
- cnn_bilstm
- cnn_gru
- cnn

Special for speech handcrafted features, you can run:
```bash
python3 training_module/handcrafted_feature_speech/cnn_bigru.py 
```

```bash
python3 training_module/handcrafted_feature_speech/cnn_bilstm.py 
```

```bash
python3 training_module/handcrafted_feature_speech/cnn_bigru.py 
```

The handcrafted-feature extraction is performed on-the-fly during training. This is done to ensure proper standardization, as some features have different value ranges and need to be scaled consistently across the dataset.

## Unimodal Detection
For unimodal detection you can run 
python3 -m training_module.unimodal_detection --config training_module/config.json

the confing.json contains any hyperparameters you can change:

```json
{
    "dataset_class": "UnimodalDataset",
    "classifier": "bigruattention",
    "embedding_filename": "audio_xslr_encoded_bigru.npy",
    "base_dir": "split_dataset",
    "device": "cpu",
    "save_pred": "results/speech_predictions.json",
    "input_size": 256,

    "hyperparameters": {
        "learning_rate": 0.005,
        "epochs": 70,
        "hidden_dim": 1024,
        "batch_size": 8
    }
}
```

**Classifiers Options**:
1. bigruattention
2. lstm_fc
3. convpoolclassifier
4. bilstm_fc 

**embedding_filename**: embeddings/features file name based on the name you saved previously

**devices**: cpu or gpu

**input size**: you can change depends on the size of the features/embeddings


## Multimodal Detections
After unimodal detection, the results are saved as JSON files (e.g., eeg_predictions.json, text_predictions.json, and speech_predictions.json). It is important to note that each JSON file contains the predictions for a single model run, based on the specific feature extraction and classifier combination used.


We would need to slightly procses them to .txt to a prefered format by running:

```bash
python3 -m results/processing_results.py results/
```
With the processed prediction files, you can perform multimodal detection using one of these fusion strategies:

**1. Majority voting (Mean)**
```bash
python3 -m multimodal_detection.majority_voting                               
```

**2. Bayesian Fusion**
```bash
python3 -m multimodal_detection.bayesian_fusion                               
```

**3. Weighted Averaging**
```bash
python3 -m multimodal_detection.weighted_averaging                            
```

The scripts for Bayesian Fusion and Weighted Averaging include user configurations to specify modalities and weights. All final results are saved in the results/ folder.


