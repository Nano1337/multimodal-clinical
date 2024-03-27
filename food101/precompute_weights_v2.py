import os
import numpy as np
from cuml.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingStats:
    def __init__(self, text_embeds, image_embeds, labels, classes):
        self.text_embeds = np.array(text_embeds)[:, 0, :]
        self.image_embeds = np.array(image_embeds)[:, 0, :]
        self.modality_embeddings = [self.text_embeds, self.image_embeds]
        self.labels = labels
        self.classes = classes
        self.num_classes = len(classes)
        self.num_samples = self.text_embeds.shape[0]
        self.num_modalities = 2  # Text and image modalities

        self.calc_distance()

    def calc_distance(self):
        print("Normalizing embeddings")
        # Normalize embeddings into unit vectors to use cosine distance
        self.modality_embeddings = [embeds / np.linalg.norm(embeds, axis=-1, keepdims=True) for embeds in self.modality_embeddings]

        # calculate the class conditional mean embedding for each modality 
        class_conditional_means = []
        for modality_embeds in self.modality_embeddings:
            means = np.zeros((self.num_classes, modality_embeds.shape[1]))
            for i, class_label in enumerate(self.classes):
                class_indices = np.where(self.labels == class_label)[0]
                print(f"Class {class_label} has {len(class_indices)} samples.")  # Debug print
                if len(class_indices) > 0:
                    class_embeds = modality_embeds[class_indices]
                    means[i] = np.mean(class_embeds, axis=0)
                else:
                    print(f"No embeddings found for class {class_label}.")  # Highlight empty classes
                class_embeds = modality_embeds[class_indices]
                means[i] = np.mean(class_embeds, axis=0)
            class_conditional_means.append(means)
        class_conditional_means = np.array(class_conditional_means)
        print(class_conditional_means.shape)
        exit()


        dim_reduced_embeds = []

        for k, modality_embeds in enumerate(self.modality_embeddings):
            print(f"Modality {k} embeddings shape before UMAP: {modality_embeds.shape}")
            modality_embeds = modality_embeds / np.linalg.norm(modality_embeds, axis=-1, keepdims=True)
            dim_reduced_embeds.append(modality_embeds)
            print(f"Modality {k} embeddings shape after UMAP: {modality_embeds.shape}")

        self.modality_embeddings = np.array(dim_reduced_embeds)



    def get_confidence_scores(self, temperature=0.7, epsilon=1e-12):
        # Vectorized computation of scaled_kmds and modality_scores
        scaled_kmds = np.array(self.kmds) / temperature
        max_scaled_kmd = np.max(scaled_kmds, axis=0, keepdims=True)
        modality_scores = np.exp(scaled_kmds - max_scaled_kmd) / (np.exp(max_scaled_kmd) + epsilon)

        confidence_scores = modality_scores.T  # Transpose to match the desired shape
        
        # Vectorized computation of calibrated_scores
        max_modality_score = np.max(confidence_scores, axis=1, keepdims=True)
        calibrated_scores = confidence_scores / (max_modality_score + epsilon)
        
        return calibrated_scores

def precompute_weights(data_path, mode):
    text_feature_path = os.path.join(data_path, "text_embed", f'{mode}_token/')
    image_feature_path = os.path.join(data_path, "image_embed", f'{mode}_imgs/')
    stat_path = os.path.join(data_path, "stat_food.txt")

    data = []
    data2class = {}
    missing_embeddings = []

    with open(os.path.join(data_path, f"my_{mode}_food.txt"), "r") as f:
        csv_reader = f.readlines()
        for single_line in csv_reader:
            item = single_line.strip().split(".jpg ")
            token_path = os.path.join(text_feature_path, item[0] + '_token.npy')
            visual_path = os.path.join(image_feature_path, item[0] + ".jpg.npy")
            if os.path.exists(token_path) and os.path.exists(visual_path):
                data.append(item[0])
                data2class[item[0]] = item[1]
            else: 
                missing_embeddings.append(item[0])

    if missing_embeddings:
        print("Samples missing embeddings:")
        for sample in missing_embeddings:
            print(sample)
    else:
        print("All samples have embeddings.")
    av_files = data

    with open(stat_path, "r") as f1:
        classes = [sclass.strip() for sclass in f1.readlines()]

    classes = sorted(classes)
    text_embeds = []
    image_embeds = []
    labels = []

    # Load precomputed embeddings in batches
    batch_size = 10000
    for start in range(0, len(av_files), batch_size):
        end = min(start + batch_size, len(av_files))
        batch_av_files = av_files[start:end]

        batch_text_embeds = []
        batch_image_embeds = []
        batch_labels = []

        for av_file in batch_av_files:
            token_path = os.path.join(text_feature_path, av_file + '_token.npy')
            visual_path = os.path.join(image_feature_path, av_file + ".jpg.npy")
            label = classes.index(data2class[av_file])

            batch_text_embeds.append(np.load(token_path))
            batch_image_embeds.append(np.load(visual_path))
            batch_labels.append(label)

        text_embeds.extend(batch_text_embeds)
        image_embeds.extend(batch_image_embeds)
        labels.extend(batch_labels)

    embedding_stats = EmbeddingStats(text_embeds, image_embeds, labels, classes)
    output = embedding_stats.get_confidence_scores()

    count_less = sum(1 for score_pair in output if score_pair[0] < score_pair[1])
    print(f"Number of instances where the first number is less than the second: {count_less}")
    
    print("Sample output weights:", output[:50])
    weights_path = os.path.join(data_path, f"weights_{mode}.npy")
    np.save(weights_path, np.array(output))   

if __name__ == "__main__":
    data_path = "../data/food101/"
    precompute_weights(data_path, mode='train')