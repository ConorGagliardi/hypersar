import numpy as np
import torch
import pandas as pd
from datetime import datetime

def precision(correct_predictions, k):
    num_hit = torch.sum(correct_predictions, dim=-1)
    return num_hit / k

def recall(correct_predictions, num_relevant):
    num_hit = torch.sum(correct_predictions, dim=-1)
    return num_hit / num_relevant

def ndcg(correct_predictions, num_relevant, k):
    ideal_correct_predictions = torch.zeros_like(correct_predictions)
    batch_size = ideal_correct_predictions.shape[0]
    for sample in range(batch_size):
        ideal_correct_predictions[sample, :num_relevant[sample]] = 1
    return dcg(correct_predictions, k) / dcg(ideal_correct_predictions, k)

def dcg(correct_predictions, k):
    result = 0.0
    for rank in range(k):
        result += correct_predictions[:, rank] / np.log2(rank + 2)
    return result

def map(correct_predictions, num_relevant, k):
    result = 0.0
    for rank in range(k):
        result += precision(correct_predictions[:, :rank + 1], rank + 1) * correct_predictions[:, rank]
    result /= num_relevant
    return result

def evaluate(correct_predicted_interactions, num_true_interactions, metrics):
    """
    Evaluates a ranking model in terms of precision and recall for the given cutoff values
    Args:
        correct_predicted_interactions: (array<bool>: n_rows * max(cutoffs)) 1 iff prediction matches a true interaction
        num_true_interactions: (array<bool>: n_rows) number of true interactions associated to each row
        metrics: (list<tuple<string,int>>) list of metrics to consider, with tuples made of the metric type and cutoff

    Returns:
        eval_results: dictionary with evaluation results for each metric cumulated over all rows; keys are the metrics
    """
    eval_results = {}
    for metric in metrics:
        (metric_type, k) = metric # Get the metric type and cutoff e.g. ("precision", 5)
        correct_predictions = correct_predicted_interactions[:, :k]
        k = min(k, correct_predictions.shape[1])
        if metric_type == "precision":
            eval_results[metric] = precision(correct_predictions, k)
        elif metric_type == "recall":
            eval_results[metric] = recall(correct_predictions, num_true_interactions)
        elif metric_type == "ndcg":
            eval_results[metric] = ndcg(correct_predictions, num_true_interactions, k)
        elif metric_type == "map":
            eval_results[metric] = map(correct_predictions, num_true_interactions, k)

    return eval_results
    
    
    
#helper functions for returning hits
def load_dictionaries(item_dict_path, tag_path):
    item_dict = {}
    with open(item_dict_path, "r", encoding='utf-8') as f:
        for line in f:
            old_id, new_id = line.strip().split("\t")
            item_dict[int(new_id)] = int(old_id)
    
    tag_df = pd.read_csv(tag_path, sep="\t", encoding="unicode_escape")
    tag_dict = {row["tagID"]: row["tagValue"] for _, row in tag_df.iterrows()}
    
    return item_dict, tag_dict
    
def load_artist_dictionary(artist_path):
    artist_df = pd.read_csv(artist_path, sep="\t", encoding="unicode_escape")
    artist_dict = {row["id"]: row["name"] for _, row in artist_df.iterrows()}
    return artist_dict
    

def map_ids_to_values(example_user_id, example_query_keywords, example_top_10_hits, item_dict, tag_dict, artist_dict):
    example_artist_ids = [item_dict[hit] for hit in example_top_10_hits]
    example_artist_names = [artist_dict[artist_id] for artist_id in example_artist_ids]
    example_query_tags = [tag_dict[keyword_id] for keyword_id in example_query_keywords]
    return example_user_id, example_query_tags, example_artist_names
    
    
    
    

def predict_evaluate(data_loader, options, model, known_interactions):
    max_k = max([metric[1] for metric in options.metrics])
    max_k = min(max_k, options.num_item)
    types = ['all', 'rec', 'search']
    eval_results = {type: {metric: torch.tensor([], dtype=torch.float, device=options.device_ops)
                           for metric in options.metrics} for type in types}

    print_example = True

    for (batch_id, batch) in enumerate(data_loader):
        if batch_id % 1 == 0:
            print("Number of batches processed: " + str(batch_id) + "...", datetime.now(), flush=True)

        device_embed = options.device_embed
        device_ops = options.device_ops
        user_ids = batch['user_ids'].to(device_embed)
        item_ids = batch['item_ids'].to(device_ops)
        interaction_types = batch['interaction_types'].to(device_ops)
        batch_size = len(user_ids)

        # Predict the items interacted for each user and mask the items which appeared in known interactions
        if options.model in ["FactorizationMachine", "DeepFM", "JSR", "DREM", "HyperSaR"]:
            keyword_ids = batch['keyword_ids'].to(device_embed)
            query_sizes = batch['query_sizes'].to(device_ops) 
            predicted_scores = model.predict(user_ids, keyword_ids, query_sizes)
        else:
            predicted_scores = model.predict(user_ids)
        ## Shape of predicted_scores: (batch_size, num_item)

        # Mask for each user the items from their training set
        mask_value = -np.inf
        for i, user in enumerate(user_ids):
            if int(user) in known_interactions:
                for interaction in known_interactions[int(user)]:
                    item = interaction[0]
                    predicted_scores[i, item] = mask_value
        _, predicted_interactions = torch.topk(predicted_scores, k=max_k, dim=1, largest=True, sorted=True)
        ## Shape of predicted_interactions: (batch_size, num_item)


        ##print an example search and retrieval

        ##match id's to real strings
        item_dict_path = "/home/stu15/s1/cgg3724/ir2023/p1c/hypersar/data/lastfm/item_dict.txt"
        tag_path = "/home/stu15/s1/cgg3724/ir2023/p1c/hypersar/data/lastfm/tags.dat"
        artist_path = "/home/stu15/s1/cgg3724/ir2023/p1c/hypersar/data/lastfm/artists.dat"
        artist_dict = load_artist_dictionary(artist_path)
        item_dict, tag_dict = load_dictionaries(item_dict_path, tag_path)


        if batch_id == 0 and print_example:
            example_user_id = user_ids[0].item()
            example_query_keywords = [keyword_id.item() for keyword_id in keyword_ids[0]].remove(-1)
            example_top_10_hits = predicted_interactions[0, :10].tolist()
            print(f"Example search for user {example_user_id} with query keywords {example_query_keywords}:")
            print(f"Top 10 hits: {example_top_10_hits}")
            example_user_id, example_query_tags, example_artist_names = map_ids_to_values(example_user_id, example_query_keywords, example_top_10_hits, item_dict, tag_dict, artist_dict)
            print(f"Example search for user {example_user_id} with query tags {example_query_tags}:")
            print(f"Top 10 hits: {example_artist_names}")
            print_example = False


        # Identify the correct interactions in the top-k predicted items
        correct_predicted_interactions = (predicted_interactions == item_ids.unsqueeze(-1)).float()
        ## Shape of correct_predicted_interactions: (batch_size, max_k)
        num_true_interactions = torch.ones([batch_size], dtype=torch.long, device=options.device_ops) # 1 relevant item
        ## Shape of num_true_interactions: (batch_size)

        # Perform the evaluation
        batch_results = {}
        batch_results['all'] = evaluate(correct_predicted_interactions, num_true_interactions, options.metrics)
        ## Separate results for recommendation and search instances
        recommendation_ids = torch.where(interaction_types == 0)[0]
        batch_results['rec'] = {metric: batch_results['all'][metric][recommendation_ids] for metric in options.metrics}
        search_ids = torch.where(interaction_types == 1)[0]
        batch_results['search'] = {metric: batch_results['all'][metric][search_ids] for metric in options.metrics}

        eval_results = {type: {metric: torch.cat((eval_results[type][metric], batch_results[type][metric]), dim=0)
                                for metric in options.metrics} for type in types}

    eval_results = {type: {metric: torch.mean(eval_results[type][metric], dim=0) for metric in options.metrics}
                    for type in types}

    return eval_results