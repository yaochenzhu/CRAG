import numpy as np
from editdistance import eval as distance


def recall_at_k(hits, num_gt, k):
    hits_at_k = hits[:k]
    return sum(hits_at_k) / num_gt


def dcg_at_k(hits, k):
    if len(hits) == 1:
        return hits[0]
    k = min(k, len(hits))
    return hits[0] + sum(hits[i] / np.log2(i + 2) for i in range(1, k))


def ndcg_at_k(hits, num_gt, k):
    idea_hits = np.zeros(len(hits), dtype=int)
    idea_hits[:num_gt] = 1
    idea_dcg = dcg_at_k(idea_hits, k)
    dcg = dcg_at_k(hits, k)
    return dcg/idea_dcg


def remove_seen(item, rec_list):
    flattened_triples = [
        (iid, title, attitude)
        for iids, titles, attitudes in zip(item["clean_context_imdb_ids"], item["clean_context_titles"], item["clean_context_attitudes"])
        for iid, title, attitude in zip(iids, titles, attitudes)
    ]
    
    # Filter titles with non-negative attitudes and ensure uniqueness
    seen_titles = {
        title for iid, title, attitude in flattened_triples
    }
    return [rec for rec in rec_list if rec not in seen_titles]


def evaluate_direct_match(item, k, K, gt_field):
    rec_list_raw = item[f'rec_list_raw_{K}']
    rec_list_raw = remove_seen(item, rec_list_raw)
    groundtruths = item[gt_field]
    
    hits = np.zeros(len(rec_list_raw), dtype=int)
    for gt in groundtruths:
        match_results = [distance(gt, rec) for rec in rec_list_raw]
        matched = False
        for i, res in enumerate(match_results):
            if res == 0 and not matched:
                hits[i] = 1
                matched = True

    num_gt = len(groundtruths)
    recall = recall_at_k(hits, num_gt, k)
    ndcg = ndcg_at_k(hits, num_gt, k)
    return recall, ndcg


def evaluate_direct_match_reflect_rerank(item, k, K, gt_field):
    rec_list_raw = item[f'rec_after_reflect_{K}']
    rec_list_raw = remove_seen(item, rec_list_raw)
    
    groundtruths = item[gt_field]
    
    hits = np.zeros(len(rec_list_raw), dtype=int)
    for gt in groundtruths:
        match_results = [distance(gt, rec) for rec in rec_list_raw]
        matched = False
        for i, res in enumerate(match_results):
            if i >= len(hits):
                break
            if res == 0 and not matched:
                hits[i] = 1
                matched = True

    num_gt = len(groundtruths)
    recall = recall_at_k(hits, num_gt, k)
    ndcg = ndcg_at_k(hits, num_gt, k)
    return recall, ndcg