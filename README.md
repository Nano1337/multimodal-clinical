# Multimodal Enfusion

### Currently functional benchmarks: 
- Audio-Video Event Localization (AVE), transformed into classification task per PMR paper
- Audio-Video MNIST (AV-MNIST) for Digit Classification
- Crema-D (Audio, Video) for Emotion Recognition
- Enrico (Screenshot, Wireframe) for Design Classification
- FakeNews (Text, Image) for Fake News Detection
- Food101 (Image, Text) for Food Classification
- MIMIC (Metadata, Timeseries) for EHR Mortality Prediction
- VGGSound (Audio, Video) for Sound Classification

### TODO: 

Tuesday:
- Update YAMLs to override a base config file
- Update every run_training.py file for top and bottom containerization in utils edits

Rest of Week:
- Update Crema-D non-base models to inherit from BaseModel too
- Update every model file to inherit from BaseModel (done with Crema-D)
- Do more HPO/read other paper configs to improve baseline perf
- Incorporate OGM-GE testing to every dataset
- Incorporate QMF testing to every dataset

## Changelog

March 19th:
- Update BaseModels with EMA for unimodal training accuracy calibration
- Log both calibrated and uncalibrated unimodal accuracies for train/val/test
- Log config file to WandB too
- Improve checkpointing logic
- Encapsulate lightning trainer logic into utils