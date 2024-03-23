import json
import os
from tqdm import tqdm


if __name__ == "__main__":

    data_dir = "/home/haoli/Documents/multimodal-clinical/data/food101/"

    all_jsonls = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    result = []
    for jsonl in all_jsonls:
        
        json_path = os.path.join(data_dir, jsonl)

        img_list = [json.loads(line)["img"].split("/")[-1] for line in open(json_path)]
        label_list = [json.loads(line)["label"] for line in open(json_path)]

        for i, img in tqdm(enumerate(img_list)):
            result.append("{} {}\n".format(img, label_list[i]))
        
        with open(os.path.join(data_dir, "my_{}_food.txt".format(jsonl.split(".jsonl")[0])), "w") as mf:
            mf.writelines(result)
        result = []