# Multimodal Enfusion

## Functional benchmarks: 
- Audio-Video Event Localization (AVE), transformed into classification task per PMR paper
- Audio-Video MNIST (AV-MNIST) for Digit Classification
- Crema-D (Audio, Video) for Emotion Recognition
- Enrico (Screenshot, Wireframe) for Design Classification
- FakeNews (Text, Image) for Fake News Detection
- Food101 (Image, Text) for Food Classification
- MIMIC (Metadata, Timeseries) for EHR Mortality Prediction
- VGGSound (Audio, Video) for Sound Classification

## Usage

For example, to run Crema-D, modify the corresponding YAML and run the following command:
```bash
python main.py --dir cremad
```



## Changelog

March 19th:
- [x] Update BaseModels with EMA for unimodal training accuracy calibration
- [x] Log both calibrated and uncalibrated unimodal accuracies for train/val/test
- [x] Log config file to WandB too
- [x] Improve checkpointing logic
- [x] Encapsulate lightning trainer logic into utils
- [x] Update YAMLs to override a base config file


### TODO: 

Rest of Week:
- Update every run_training.py file for config and trainer encapsulation
- Update Crema-D non-base models to inherit from BaseModel too
- Update every model file to inherit from BaseModel (done with Crema-D)
- Do more HPO/read other paper configs to improve baseline perf
- Incorporate OGM-GE testing to every dataset
- Incorporate QMF testing to every dataset

Haoli's Personal Stuff: 
- Clean up useless ckpts in each subfolder since they're all supposed to be in the data folder now