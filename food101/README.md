# Food101 Setup Steps: 

1. Download UPMC Food101 from the [Kaggle Dataset](https://www.kaggle.com/datasets/gianmarco96/upmcfood101), put it in the data folder, unzip it and rename the directory to `food101` in the `data` parent folder.
2. Download each of the `*.jsonl` files from [this link](https://github.com/QingyangZhang/QMF/tree/cccee6fb667266b6ac74356cce38aacd75e00540/text-image-classification/datasets/food101) and put it under the `data/food101` directory
3. Copy [stat_food.txt](https://github.com/Cecile-hi/Multimodal-Learning-with-Alternating-Unimodal-Adaptation/blob/main/data/stat_food.txt) to the `data/food101` directory. 
4. In `multimodal-enfusion/food101`, use the `gen_food_txt.py` script to generate data path / label pairs. Make sure you change the `data_dir` variable to point to the `data/food101` directory. 
5. In `multimodal-enfusion/food101`, use the `extract_token.py` script to preprocess data. Make sure you change the `json_dir` variable to point to the `data/food101` directory. This script will create the train, val (called dev in this dataset), and test dataset directories. 