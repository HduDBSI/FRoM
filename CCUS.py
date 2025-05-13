import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
from CustomMetrics import cal_metrics
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import random

class TextCNN(nn.Module):
    def __init__(self, vocab_size=50265, embed_dim=768, class_num=4, filter_size_list=[1, 2, 3, 4, 5], filter_num=128):
        """
        Args:
            vocab_size (int): Size of vacabularies
            class_num (int): Number of output classes.
            filter_size_list (list): List of filter sizes for convolutional layers.
            filter_num (int): Number of filters per convolutional layer.
        """
        super(TextCNN, self).__init__()

        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, filter_num, (filter_size, embed_dim)) for filter_size in filter_size_list])
        self.fc = nn.Linear(filter_num * len(filter_size_list), class_num)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, class_num).
        """
        x = self.embed_layer(x) # (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)

        # Add a channel dimension for convolutional layers
        x = x.unsqueeze(1)  # (batch_size, seq_len, emb_dim) -> (batch_size, 1, seq_len, embed_dim)

        # Apply convolutional layers and max pooling
        conv_outs = [F.relu(conv(x).squeeze(3)) for conv in self.conv_layers]

        pooled_outs = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outs]

        # Concatenate pooled outputs
        embed = torch.cat(pooled_outs, dim=1)  # (batch_size, filter_num * len(filter_size_list))
        x = self.fc(embed)  # (batch_size, class_num)

        return embed, x

# CNN_CosineDistance_Under_Sampling
class CCUS:
    def __init__(self, device, model_name, class_num, train_epoch=20, threshold=0.5, batch_size=1024, lr=1e-3, seed=42):
        self.model = TextCNN(class_num=class_num).to(device)
        self.device = device
        self.threshold = threshold
        self.train_epoch = train_epoch
        self.model_name = model_name
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        random.seed(seed)

    def fit_resample(self, train_set, valid_set):
        self.train_model(train_set, valid_set)
        neg_easy_embed, neg_easy_idx, other_idx = self.filter_samples(train_set)
        neg_easy_rep_idx = self.cluster_easy_negatives(neg_easy_idx, neg_easy_embed)
        
        new_idx = np.concatenate([neg_easy_rep_idx, other_idx])
        new_dataset = Subset(train_set, new_idx)
        print('org dataset size:', len(train_set))
        print('new dataset size:', len(new_dataset))
        return new_dataset

    def filter_samples(self, dataset):
        self.load_model()
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        neg_idx, neg_prob, neg_embed = [], [], []
        with torch.no_grad():
            for inputs in dataloader:
                tokens = inputs['input_ids'].to(self.device)
                embed, x = self.model(tokens)
                prob = torch.softmax(x, dim=1)

                neg_idx.append(inputs['idx'][inputs['label'] == 0].numpy())
                neg_prob.append(prob[inputs['label'] == 0, 0].cpu().numpy().astype('float16'))
                neg_embed.append(embed[inputs['label'] == 0].cpu().numpy().astype('float16'))

        self.clear_memory()

        neg_idx = np.concatenate(neg_idx)
        neg_prob = np.concatenate(neg_prob)
        neg_embed = np.vstack(neg_embed) 

        prob = np.percentile(neg_prob, (1-self.threshold)*100)

        neg_easy_idx = neg_idx[neg_prob >= prob]
        neg_easy_embed = neg_embed[neg_prob >= prob]

        other_idx = np.array(list(set(range(len(dataset))) - set(neg_easy_idx))) # include neg_easy_idx and pos_all
        return neg_easy_embed, neg_easy_idx, other_idx

    def cluster_easy_negatives(self, neg_easy_idx, neg_easy_embed):
        rep_points = []
        remain_points = set(range(neg_easy_idx.size))  # 用集合管理剩余点索引

        similarities = cosine_similarity(neg_easy_embed)
        np.fill_diagonal(similarities, 0)  # 避免选中自身

        while remain_points:
            this_point = remain_points.pop()
            rep_points.append(this_point)  # 当前点作为代表点

            # 找到与当前点相似度 >= 0.99 的所有点
            similar_points = list(np.where(similarities[this_point, :] >= 0.99)[0])
            
            # 从相似度矩阵中移除这些点
            similarities[:, similar_points] = 0
            similarities[similar_points, :] = 0

            # 从剩余点集合中移除这些点
            remain_points.difference_update(similar_points)  

            # 找到与当前点相似度 >= 0.95 的所有点
            similar_points = list(np.where(similarities[this_point, :] >= 0.95)[0])
            sampled_points = maximize_diversity(similar_points, similarities, len(similar_points) // 2)
            # sampled_points = random.sample(similar_points, 2 * len(similar_points) // 3)

            rep_points.extend(sampled_points)

            similarities[:, similar_points] = 0
            similarities[similar_points, :] = 0
            remain_points.difference_update(similar_points)  

        neg_easy_rep_idx = neg_easy_idx[rep_points]

        print("CCUS: Clustering finished")
        print(f"Removed: {neg_easy_idx.size-neg_easy_rep_idx.size}")
        return neg_easy_rep_idx

    def train_model(self, train_set, valid_set):
        pbar = tqdm(range(self.train_epoch), total=self.train_epoch, ncols=100, unit="epoch", colour="red")
        best_f1 = 0.2
        for i in pbar:
            self.train_one_epoch(train_set)
            valid_f1 = self.evaluate(valid_set)
            if best_f1 < valid_f1:
                best_f1 = valid_f1
                self.save_model()
 
        print('CCUS: CNN model training finished')

    def save_model(self):
        os.makedirs('models/', exist_ok=True)
        model = {'textcnn': self.model.state_dict()}
        torch.save(model, 'models/'+self.model_name)
    
    def load_model(self):
        content = torch.load('models/'+self.model_name, map_location=lambda storage, loc:storage)
        self.model.load_state_dict(content['textcnn'])

    def train_one_epoch(self, train_set):    
        dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        total_loss = 0
        for inputs in dataloader:
            self.optimizer.zero_grad()

            input_ids = inputs['input_ids'].to(self.device)
            label = inputs['label'].to(self.device)

            _, logit = self.model(input_ids)
            loss = loss_fn(logit, label)
        
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        total_loss = total_loss / len(list(dataloader))
        return total_loss

    def evaluate(self, valid_set):
        dataloader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False)
        y_pred, y_true, y_pred_logit = [], [], []
        
        self.model.eval()
        for inputs in dataloader:
            tokens = inputs['input_ids'].to(self.device)

            with torch.no_grad():
                _, logit = self.model(tokens)

                prob = torch.softmax(logit, dim=1)
                tmp_pred = prob.argmax(dim=1).int().cpu()
                y_pred_positive_prob = (1 - prob[:,0]).cpu()

                y_pred += tmp_pred
                y_true += inputs['label'].tolist()
                y_pred_logit += y_pred_positive_prob

        metrics = cal_metrics(y_true, y_pred, y_pred_logit)

        return metrics['MacroF'] if 'MacroF' in metrics else metrics['F']
    
    # Clears the memory used by the model and optimizer to free GPU resources.
    def clear_memory(self):
        if self.model is not None:
            del self.model
        if self.optimizer is not None:
            del self.optimizer
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

# def maximize_diversity(indices, S, k):
#     m = len(indices)  # 点的数量
#     # 建立编号到矩阵索引的映射
#     index_map = {idx: i for i, idx in enumerate(indices)}
    
#     # dp[i][j] 表示从前 i 个点中选出 j 个点的最小相似度和
#     dp = [[float('inf')] * (k + 1) for _ in range(m + 1)]
#     selected_points = [[None] * (k + 1) for _ in range(m + 1)]
    
#     # 初始化
#     for i in range(m + 1):
#         dp[i][0] = 0  # 选 0 个点时相似度和为 0

#     # 动态规划求解
#     for j in range(1, k + 1):  # 枚举选取点数
#         for i in range(1, m + 1):  # 枚举点数范围
#             for x in range(i):  # 枚举前一个选中点的位置
#                 # 获取当前点和已选点集合的相似度和
#                 similarity_sum = sum(S[index_map[indices[i-1]]][index_map[indices[p]]] for p in (selected_points[x][j-1] or []))
#                 if dp[x][j-1] + similarity_sum < dp[i][j]:
#                     dp[i][j] = dp[x][j-1] + similarity_sum
#                     selected_points[i][j] = (selected_points[x][j-1] or []) + [i-1]
    
#     # 结果
#     min_similarity = float('inf')
#     best_set = None
#     for i in range(1, m + 1):
#         if dp[i][k] < min_similarity:
#             min_similarity = dp[i][k]
#             best_set = selected_points[i][k]
    
#     # 将索引映射回原始点编号
#     if best_set is not None:
#         best_set = [indices[i] for i in best_set]
#     else:
#         best_set = []
    
#     return best_set, min_similarity

def maximize_diversity(points, similarities, num_points):
    """
    Greedy algorithm to select 'num_points' from the available points, 
    minimizing the sum of their similarities (maximizing diversity).
    Each selected point is the one that has the minimum similarity with 
    the previously selected points.
    """
    if len(points) == 0:
        return []
    
    selected = set()
    available = set(points)  # Set of all available points to choose from

    selected.add(points[0])
    available.remove(points[0])

    while len(selected) < num_points and available:
        # For each candidate point, compute its total similarity with the already selected points
        best_point = None  # Placeholder for the point with the minimum similarity sum
        min_similarity = float('inf')  # Initialize to a very large value

        for point in available:
            similarity_sum = sum(similarities[point, s] for s in selected)

            # If the current point has a lower similarity sum (i.e., higher diversity), select it
            if similarity_sum  < min_similarity:
                min_similarity = similarity_sum
                best_point = point

        # Add the selected point to the list of selected points
        if best_point is not None:
            selected.add(best_point)
            available.remove(best_point)

    return list(selected)

    # def cluster_easy_negatives(self, neg_easy_idx, neg_easy_embed):
    #     n_clusters = max(int(neg_easy_idx.size * self.cluster_percent / 10), 200)
    #     print('cluster num:', n_clusters)
    #     kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, batch_size=self.batch_size)
    #     cluster_labels = kmeans.fit_predict(neg_easy_embed)
        
    #     neg_easy_rep_idx = []
    #     for cluster_id in range(n_clusters):
    #         cluster_points = np.where(cluster_labels == cluster_id)[0]
    #         if len(cluster_points) == 0:
    #             continue
    #         cluster_embeds = neg_easy_embed[cluster_points]
    #         pairwise_distances = np.linalg.norm(cluster_embeds[:, None] - cluster_embeds[None, :], axis=-1)

    #         # Iteratively select points to maximize distance
    #         selected_points = []

    #         # Step 1: Select the point with the highest total distance as the first point
    #         total_distances = pairwise_distances.sum(axis=1)
    #         first_point = np.argmax(total_distances)
    #         selected_points.append(first_point)

    #         # Step 2: Iteratively select the point farthest from the last selected point
    #         for _ in range(min(len(cluster_points), 100) - 1):
    #             last_selected_point = selected_points[-1]
    #             distances_from_last = pairwise_distances[last_selected_point]
    #             next_point = np.argmax(distances_from_last)
    #             selected_points.append(next_point)

    #         # Map selected points back to original indices
    #         for point in selected_points:
    #             neg_easy_rep_idx.append(neg_easy_idx[cluster_points[point]])

    #     print('CKUS: KMeans clustering finished')
    #     return neg_easy_rep_idx