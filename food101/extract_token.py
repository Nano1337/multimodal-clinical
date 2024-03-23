"""
Credits to https://github.com/Cecile-hi/Multimodal-Learning-with-Alternating-Unimodal-Adaptation/blob/main/data/extract_token.py
"""


import torch 
import transformers
import numpy as np
import os
import json
from tqdm import tqdm

if __name__ == "__main__":
    
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    
    json_dir = "/home/haoli/Documents/multimodal-clinical/data/food101/"

    text_target_dir = os.path.join(json_dir, "text_token")
    
    # img_source_dir = os.path.join(json_dir, "data")
    img_target_dir = os.path.join(json_dir, "visual")

    # Ensure the directory exists
    os.makedirs(text_target_dir, exist_ok=True)
    os.makedirs(img_target_dir, exist_ok=True)


    all_jsonls = ["train.jsonl", "dev.jsonl", "test.jsonl"]

    for filename in all_jsonls:

        json_path = os.path.join(json_dir, filename)

        data_texts = [json.loads(line)["text"] for line in open(json_path)]
        # name_list = [json.loads(line)["id"] for line in open(json_path)]
        img_list = [json.loads(line)["img"] for line in open(json_path)]

        print("{} has {} files".format(filename, len(data_texts)))
        datasub = filename.split(".jsonl")[0]
        
        for cap_index, caption in tqdm(enumerate(data_texts)):
            encoded_caption = tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="np",
                add_special_tokens=False,
            )
            # import pdb
            # pdb.set_trace()
            tokenized_caption = torch.from_numpy(encoded_caption["input_ids"][0])[None, ...]
            padding_mask = 1.0 - encoded_caption["attention_mask"][0].astype(np.float32)
            padding_mask = torch.from_numpy(padding_mask[None, ...])

            spec_name = img_list[cap_index].split("/")[-1].split(".jpg")[0]

            tmp_text_dir = os.path.join(text_target_dir, "{}_token".format(datasub))
            tmp_image_dir = os.path.join(img_target_dir, "{}_imgs".format(datasub))

            os.makedirs(tmp_text_dir, exist_ok=True)
            os.makedirs(tmp_image_dir, exist_ok=True)

            token_save_path = os.path.join(tmp_text_dir, 
                                           "{}_token.npy".format(spec_name))
            pm_save_path = os.path.join(tmp_text_dir, 
                                        "{}_pm.npy".format(spec_name))
            img_source = os.path.join(json_dir, img_list[cap_index])
            img_target = os.path.join(tmp_image_dir, img_list[cap_index].split("/")[-1])
        
            np.save(token_save_path, np.array(tokenized_caption))
            np.save(pm_save_path, np.array(padding_mask))
            
            os.system("cp {} {}".format(img_source, img_target))
    print("Done!")