# Multimodal Enfusion

By: Haoli Yin and Jenni Crawford

## Introduction

A Work in Progress...

## Functional benchmarks: 
- [ ] Audio-Video Event Localization (AVE), transformed into classification task per PMR paper
- [ ] Audio-Video MNIST (AV-MNIST) for Digit Classification
- [X] Crema-D (Audio, Video) for Emotion Recognition
- [ ] Enrico (Screenshot, Wireframe) for Design Classification
- [ ] FakeNews (Text, Image) for Fake News Detection
- [ ] Food101 (Image, Text) for Food Classification
- [ ] MIMIC (Metadata, Timeseries) for EHR Mortality Prediction
- [ ] VGGSound (Audio, Video) for Sound Classification

## Setup

To get started, create a python virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate
```

Optionally, install the fast uv installer for faster installation
```bash
pip install uv
uv pip install -r requirements.txt
```

Or install the requirements directly
```bash
pip install -r requirements.txt
```

## Usage

For example, to run Crema-D, modify the corresponding YAML and run the following command:
```bash
python main.py --dir cremad
```

## Changelog

March 21st: 
- [X] - Update Crema-D non-base models to inherit from BaseModel too (move other-works specific to another dir other than utils)

March 19th:
- [x] Update BaseModels with EMA for unimodal training accuracy calibration
- [x] Log both calibrated and uncalibrated unimodal accuracies for train/val/test
- [x] Log config file to WandB too
- [x] Improve checkpointing logic
- [x] Encapsulate lightning trainer logic into utils
- [x] Update YAMLs to override a base config file


### TODO: 

Rest of Week:

- Implement QMF for Crema-D
- Update every run_training.py file for config and trainer encapsulation
- Update every model file to inherit from BaseModel (done with Crema-D)
- Do more HPO/read other paper configs to improve baseline perf
- Incorporate OGM-GE testing to every dataset
- Incorporate QMF testing to every dataset
- Create inference script with config to run testing only

Haoli's Personal Stuff: 
- Clean up useless ckpts in each subfolder since they're all supposed to be in the data folder now

