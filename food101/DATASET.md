## Shortcut

For ease of access, I will provide the the preprocessed dataset (Google Drive)[https://drive.google.com/file/d/1S0Oz72Y28fTijnsR6RZxNqPrnaKKrCwT/view?pli=1] link as long as my Google Account is active. The Crema-D dataset is made available under the Open Database License so this should be fine (let me know if it's not). You can use the following commands to download the dataset:

```bash
gdown --id 1S0Oz72Y28fTijnsR6RZxNqPrnaKKrCwT
pigz -d crema-d.tar.gz
tar -xvf crema-d.tar
```

## Long way
Manual preprocessing steps from scratch are the exact same as that found in DATASET.md in the VGGSound directory. You can download the raw data via the [original repo](https://github.com/CheyneyComputerScience/CREMA-D) that uses git lfs 