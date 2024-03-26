import os
import numpy as np
from scipy.stats import multivariate_normal
from cuml import UMAP
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

        self.reduce_dim()

    

    def reduce_dim(self):
        # Normalize embeddings into unit vectors to use cosine distance
        self.modality_embeddings = [embeds / np.linalg.norm(embeds, axis=-1, keepdims=True) for embeds in self.modality_embeddings]
        all_embeds = np.concatenate(self.modality_embeddings)

        # Perform UMAP dimensionality reduction
        reducer = UMAP(n_components=160, metric="cosine")
        reducer.fit(all_embeds)
        dim_reduced_embeds = []

        for k, modality_embeds in enumerate(self.modality_embeddings):
            print(f"Modality {k} embeddings shape before UMAP: {modality_embeds.shape}")
            modality_embeds = reducer.transform(modality_embeds.reshape(modality_embeds.shape[0], -1))
            modality_embeds = modality_embeds / np.linalg.norm(modality_embeds, axis=-1, keepdims=True)
            dim_reduced_embeds.append(modality_embeds)
            print(f"Modality {k} embeddings shape after UMAP: {modality_embeds.shape}")

        self.modality_embeddings = np.array(dim_reduced_embeds)
        self.calculate_kmeans()

    def calculate_kmeans(self):
        self.kmds = []

        kmeans = KMeans(n_clusters=self.num_classes, random_state=42)
        for modality_embeds in self.modality_embeddings:
            output = kmeans.fit(modality_embeds)
            centroids = output.cluster_centers_
            similarity = cosine_similarity(modality_embeds, centroids)
            closest = np.argmax(similarity, axis=1)
            distances = np.array([1 - similarity[i, closest[i]] for i in range(len(modality_embeds))])
            self.kmds.append(distances)

    def get_confidence_scores(self, temperature=1.0, epsilon=1e-12):
        confidence_scores = []
        calibrated_scores = []
        for kmds in self.kmds:
            scaled_kmds = [kmd / temperature for kmd in kmds]
            max_scaled_kmd = np.max(scaled_kmds)
            modality_scores = [np.exp(kmd - max_scaled_kmd) / (np.exp(max_scaled_kmd) + epsilon) for kmd in scaled_kmds]
            confidence_scores.append(modality_scores)

        # Calibration step
        for i in range(len(self.text_embeds)):
            max_modality_score = max(confidence_scores[k][i] for k in range(self.num_modalities))
            calibrated_scores.append([score / (max_modality_score + epsilon) for score in [confidence_scores[k][i] for k in range(self.num_modalities)]])

        return calibrated_scores

def precompute_weights(data_path, mode):
    text_feature_path = os.path.join(data_path, "text_embed", f'{mode}_token/')
    image_feature_path = os.path.join(data_path, "image_embed", f'{mode}_imgs/')
    stat_path = os.path.join(data_path, "stat_food.txt")

    data = []
    data2class = {}

    with open(os.path.join(data_path, f"my_{mode}_food.txt"), "r") as f:
        csv_reader = f.readlines()
        for single_line in csv_reader:
            item = single_line.strip().split(".jpg ")
            token_path = os.path.join(text_feature_path, item[0] + '_token.npy')
            visual_path = os.path.join(image_feature_path, item[0] + ".jpg.npy")
            if os.path.exists(token_path) and os.path.exists(visual_path):
                data.append(item[0])
                data2class[item[0]] = item[1]

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

    print("Sample output weights:", output[:100])
    exit()
    weights_path = os.path.join(data_path, f"weights_{mode}.npy")
    np.save(weights_path, np.array(output))

if __name__ == "__main__":
    data_path = "../data/food101/"
    precompute_weights(data_path, mode='train')