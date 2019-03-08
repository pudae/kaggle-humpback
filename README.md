# kaggle-humpback-submission
Code for 3rd place solution in Kaggle Humpback Whale Identification Challange.

To read the detailed solution, please, refer to [the Kaggle post](https://www.kaggle.com/c/humpback-whale-identification/discussion/82484)

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
- 2x NVIDIA 1080 Ti

## Reproducing Submission
To reproduce my submission without retraining, do the following steps:
1. [Installation](#installation)
2. [Download Dataset](#download-dataset)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n humpback python=3.6
source activate humpback
pip install -r requirements.txt
```

## Download dataset
Download and extract *train.zip* and *test.zip* to *data* directory.
If the Kaggle API is installed, run following command.
```
$ kaggle competitions download -c humpback-whale-identification -f train.zip
$ kaggle competitions download -c humpback-whale-identification -f test.zip
$ unzip train.zip -d data/train
$ unzip test.zip -d data/test
```

### Generate CSV files
*You can skip this step. All CSV files are prepared in the data directory.*

#### List of CSV files
filename | description
------------ | -------------
landmark.{split}.{fold}.csv | predicted landmarks for the train and test set
duplcate_ids.csv | list of duplicate identities
leaks.csv | leaks from [post](https://www.kaggle.com/c/humpback-whale-identification/discussion/80086)
split.keypoint.{fold}.csv | labels for training bounding box and landmark detector
train.v2.csv | label file that duplicate ids are grouped to single identity and several new whales are also grouped. 

#### Landmark
To inference landmarks, run following commands
```
$ sh inference_landmarks.sh
```

## Training
In the configs directory, you can find configurations I used to train my final models. 

### Train models
To train models, run following commands.
```
$ python train.py --config={config_path}
```

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
densenet121 | 1x 1080 Ti | 320 | 300 | 60 hours

### Average weights
To average weights, run following commands.
```
$ python swa.py --config={config_path}
```

The averages weights will be located in *train_logs/{train_dir}/checkpoint*.

### Pretrained models
You can download pretrained model that used for my submission from [link](https://www.dropbox.com/s/fdnh29pjk8rpxgs/train_logs.zip?dl=0). Or run following command.
```
$ wget https://www.dropbox.com/s/fdnh29pjk8rpxgs/train_logs.zip
$ tar xzvf train_logs.tar.gz
```
Unzip them into train_logs then you can see the following structure:
```
results
  +- densenet121.1st
  |  +- checkpoint
  +- densenet121.2nd
  |  +- checkpoint
  +- densenet121.3rd
  |  +- checkpoint
  +- landmark.0
  |  +- checkpoint
  +- landmark.1
  |  +- checkpoint
  +- landmark.2
  |  +- checkpoint
  +- landmark.3
  |  +- checkpoint
  +- landmark.4
  |  +- checkpoint
```

## Inference
If trained weights are prepared, you can create files that contain cosine similarities of images with target whales.
```
$ python inference.py \
  --config={config_filepath} \
  --tta_landmark={0 or 1} \
  --tta_flip={0 or 1} \
  --output={output_filepath}
```
To make submission, you must inference test and test_val splits. For example:
```
$ python make_submission.py \
  --input_path={comma seperated list of similarity file paths} \
  --output_path={submission_file_path}
```
To inference all models and make submission using pretrained models, simply run `sh inference.sh`

## Post Processing
As you know, there are some duplicate whale ids. For the duplicate ids, the following process are applied.

Assume that the identity A and the identity B are duplicate. 

1. If top1 prediction is the identity A, then I set the identity B to top2 prediction.
2. If the size of test image is equal to one of images in identity A and is not equal to any of images in identity B, then I set top1 prediction to identity A. 

