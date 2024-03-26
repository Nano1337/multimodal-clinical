
import torch 
import transformers
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch


if __name__ == "__main__":
    
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    json_dir = "/home/haoli/Documents/multimodal-clinical/data/food101/"

    text_target_dir = os.path.join(json_dir, "text_embed")
    
    # img_source_dir = os.path.join(json_dir, "data")
    img_target_dir = os.path.join(json_dir, "image_embed")

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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for cap_index, caption in tqdm(enumerate(data_texts)):
            with torch.no_grad():
                model = model.to(device)

            img_source = os.path.join(json_dir, img_list[cap_index])
            image = Image.open(img_source)

            inputs = processor(text=caption, images=image, padding="max_length", return_tensors="pt", truncation=True)

            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(**inputs)
            text_embeds = output['text_embeds']
            image_embeds = output['image_embeds']

            spec_name = img_list[cap_index].split("/")[-1].split(".jpg")[0]

            tmp_text_dir = os.path.join(text_target_dir, "{}_token".format(datasub))
            tmp_image_dir = os.path.join(img_target_dir, "{}_imgs".format(datasub))

            os.makedirs(tmp_text_dir, exist_ok=True)
            os.makedirs(tmp_image_dir, exist_ok=True)

            token_save_path = os.path.join(tmp_text_dir, 
                                           "{}_token.npy".format(spec_name))
            img_target = os.path.join(tmp_image_dir, img_list[cap_index].split("/")[-1])
        
            text_embeds = text_embeds.cpu().detach().numpy()
            image_embeds = image_embeds.cpu().detach().numpy()

            np.save(token_save_path, text_embeds)
            np.save(img_target, image_embeds)

    print("Done!")