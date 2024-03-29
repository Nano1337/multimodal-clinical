import os
import numpy as np
from cuml.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
class EmbeddingStats:
    def __init__(self, text_embeds, image_embeds, labels, classes):
        self.text_embeds = np.array(text_embeds)[:, 0, :]
        self.image_embeds = np.array(image_embeds)[:, 0, :]
        self.modality_embeddings = [self.text_embeds, self.image_embeds]
        self.labels = np.array(labels)
        self.classes = classes
        self.num_classes = len(classes)
        self.num_samples = self.text_embeds.shape[0]
        self.num_modalities = 2  # Text and image modalities

        # hyperparameter
        self.alpha = 3.0


        self.calc_distance()

    def calc_distance(self):
        print("Normalizing embeddings")
        # Normalize embeddings into unit vectors to use cosine distance
        self.modality_embeddings = [embeds / np.linalg.norm(embeds, axis=-1, keepdims=True) for embeds in self.modality_embeddings]
        modality_embeds = np.stack(self.modality_embeddings)
        print(f"Modality embeddings shape: {modality_embeds.shape}")
        print(f"Labels shape: {self.labels.shape}")

        # calculate the class conditional mean embedding for each modality 
        class_cond_means = np.zeros((self.num_modalities, self.num_classes, modality_embeds.shape[-1]))
        for i in range(self.num_modalities):
            for j in range(self.num_classes):
                class_cond_means[i, j] = np.mean(modality_embeds[i, self.labels == j], axis=0)

        # calculate the cosine distance between each embedding and the class conditional mean
        # TODO: can definitely make this code more vectorized
        self.kmds = np.zeros((self.num_modalities, self.num_samples))
        for i in range(self.num_modalities):
            unimodal_class_cond_means = class_cond_means[i]
            for j in range(self.num_classes):   
                class_modality_embeds = modality_embeds[i, self.labels == j]
                unimodal_class_mean = unimodal_class_cond_means[j]
                # calculate cosine sim and put into kmds at original sample index
                cos_sim = cosine_similarity(class_modality_embeds, unimodal_class_mean.reshape(1, -1))
                self.kmds[i, self.labels == j] = 1 - cos_sim.flatten()

        self.kmds = self.kmds.T

        self.kmds = 1 - np.exp(-self.alpha * self.kmds)

        np.save("weights.npy", self.kmds)


        print(self.kmds[:100])
        exit()
        # visualize kmds scatter plots per class
        # for i in range(self.num_classes): 
        #     self.plot_scatter(i)

        self.print_remaining_samples_per_class()

    def print_remaining_samples_per_class(self):
        threshold_modality_1 = 0.8
        threshold_modality_2 = 0.75

        # Apply thresholding
        modality_1_filtered = self.kmds[:, 0] < threshold_modality_1
        modality_2_filtered = self.kmds[:, 1] < threshold_modality_2

        # Combine filters for both modalities
        combined_filter = (modality_1_filtered & modality_2_filtered).astype(int)
        # Count remaining samples per class

        # save combined_filter as npy 
        np.save("combined_filter.npy", combined_filter)

        print("Remaining samples per class after thresholding:")
        for i, class_label in enumerate(self.classes):
            filtered_classes = combined_filter[self.labels == i]
            class_num = filtered_classes.shape[0]
            filtered_num = np.sum(filtered_classes)
            print(f"Class {class_label}: {filtered_num}/{class_num} samples")

        exit()

    def plot_scatter(self, class_index=100):
        import matplotlib.pyplot as plt
        plt.figure()  # Ensure a new figure is created for each method call

        if class_index is not None:
            mask = self.labels == class_index
            kmds_filtered = self.kmds[mask]
            labels_filtered = self.labels[mask]
        else:
            kmds_filtered = self.kmds
            labels_filtered = self.labels

        # Scatter plot of kmds_filtered values colored by labels_filtered
        plt.scatter(kmds_filtered[:, 0], kmds_filtered[:, 1], c=labels_filtered, cmap='viridis')

        plt.xlabel('Text Modality Distance')
        plt.ylabel('Image Modality Distance')
        title = 'Scatter Plot of KMDs by Label'
        if class_index is not None:
            title += f' for Class {class_index}'
        plt.title(title)
        plt.colorbar(label='Label')
        plt.savefig(f'figs/scatter_plot_kmds_by_label{"_class_" + str(class_index) if class_index is not None else ""}.png')
        plt.close()  # Close the figure to free up memory


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
    # output = embedding_stats.get_confidence_scores()

    # count_less = sum(1 for score_pair in output if score_pair[0] < score_pair[1])
    # print(f"Number of instances where the first number is less than the second: {count_less}")
    
    # print("Sample output weights:", output[:50])
    # exit()
    # weights_path = os.path.join(data_path, f"weights_{mode}.npy")
    # np.save(weights_path, np.array(output))   

if __name__ == "__main__":
    data_path = "../data/food101/"
    precompute_weights(data_path, mode='train')