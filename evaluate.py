import os
import time
import json
import pickle
import threading
import multiprocessing

from tqdm import tqdm
from pprint import pprint
from copy import deepcopy
from functools import partial
from collections import defaultdict

import openai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from editdistance import eval as distance

from libs.utils import extract_movie_name, process_item_raw
from libs.utils import process_retrieval_reflect_raw
from libs.utils import process_rec_reflect_raw
from libs.model import cf_retrieve, get_response
from libs.metrics import evaluate_direct_match
from libs.metrics import evaluate_direct_match_reflect_rerank

import pdb

external = False

if external:
    ### get the openai api key from the environment
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    
    ### set organization
    if os.environ.get('OPENAI_ORG') is not None:
        openai.organization = os.environ.get('OPENAI_ORG')
    
    ### set openai.api_key
    if openai.api_key is None:
        raise Exception('OPENAI_API_KEY is not set')
else:
    import nflx_copilot as ncp
    openai = ncp
    ncp.project_id = "reddit"

### some path information
model = "gpt-4o"
method = "CRAG"
dataset = "reddit"
version = "_with_titles"
prompt_type = "rag"
datafile = f"test_clean{version}"

data_root = f"data/{dataset}"
from_pkl = f"{data_root}/{datafile}.pkl"

cf_model = "large_pop_adj_07sym"
cf_root = f"models/cf/{cf_model}"

### some hyperparameters
temperature = 0.0
max_tokens = 512
n_threads = 500
n_print = 100
tier=30
n_samples = -1
K_list = list(range(5, 40, 5))
k_list = [5,10,15,20]
metrics = {}
avg_metrics = {}

### prompts used in the CRAG model
prompt_reflect_titles = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system), "
    "as well as some movies retrieved from our movie database based on the similarity with the movies mentioned by the user in the context."
    "You need to judge whether each retrieved movie is a good recommendation based on the context.\n"
    "Here is the conversation: {context}\n"
    "Here are retrieved movies: {retrieved_titles}.\n"
    "You need to reply with the judgement of each movie in a line, in the form of movie_name####judgment, "
    "where judgement is a binary number 0, 1. Judgment 0 means the movie is a bad recommendation, whereas judgment 1 means the movie is a good recommendation. "
    "System:"
)
    
prompt_with_retrieved_titles = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system). "
    "Based on the conversation, you need to reply with 20 movie recommendations without extra sentences. "
    "List the standardized title of each movie on a separate line.\n"
    "Here is the conversation: {context}\n"
    "Based on movies mentioned in the conversation, here are some movies that are usually liked by other users: {retrieved_titles}.\n"
    "Use the above information at your discretion (i.e., do not confine your recommendation to the above movies). "
    "System:"
)

prompt_no_title = ("Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system). "
    "Based on the conversation, you need to reply with 20 recommendations without extra sentences. "
    "List the standardized title of each movie in each line.\n"
    "Here is the conversation: {context}\n"
    "System:"
)

prompt_reflect_rec_titles = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system), "
    "as well as some movie candidates from our movie database. "
    "You need to rate each retrieved movie as recommendations into five levels based on the conversation: "
    "2 (great), 1 (good), 0 (normal), -1 (not good), -2 (bad).\n"
    "Here is the conversation: {context}\n"
    "Here are the movie candidates: {rec_titles}.\n"
    "You need to reply with the rating of each movie in a line, in the form of movie_name####rating, "
    "where rating should be an Integer, and 2 means great, 1 means good, 0 means normal, -1 means not good, and -2 means bad.\n"
    "System:"
)


def load_and_process_cf_model():
    '''
        This function loads and process the item-item 
        similarity matrix for collaborative retrieval
    '''
    sim_mat_pkl = os.path.join(cf_root, "BBsim.pickle")
    with open(sim_mat_pkl, "rb") as f:
        sim_mat = pickle.load(f)

    row2imdb_id_pkl = os.path.join(cf_root, "imdb_ids.pickle")
    with open(row2imdb_id_pkl, "rb") as f:
        raw_row2imdb_id = pickle.load(f)

    raw_imdb_id2row = {iid:i for i, iid in enumerate(raw_row2imdb_id)}
    raw_row2imdb_id = {i:iid for i, iid in enumerate(raw_imdb_id2row)}

    raw_imdb_id2col = deepcopy(raw_imdb_id2row)
    raw_col2imdb_id = deepcopy(raw_row2imdb_id)
    
    ### load the reddit test context movie database
    reddit_test_meta_pkl = os.path.join(f'{data_root}/entity2id{version}.pkl')
    with open(reddit_test_meta_pkl, "rb") as f:
        reddit_test_id_name_table = pickle.load(f)

    ### process the row of the sim matrix
    reddit_test_name2id = reddit_test_id_name_table.set_index('title')['imdb_id'].to_dict()
    reddit_test_id2name = {v: extract_movie_name(k) for k, v in reddit_test_name2id.items()}    
    relevant_indices = [row for row, imdb_id in raw_row2imdb_id.items() if imdb_id in reddit_test_id2name]
    sim_mat = sim_mat[relevant_indices, :]
    row2imdb_id = {new_row: raw_row2imdb_id[old_row] for new_row, old_row in enumerate(relevant_indices)}
    imdb_id2row = {imdb_id: row for row, imdb_id in row2imdb_id.items()}
    
    ### process the column of the sim matrix
    reddit_test_resp_meta_pkl = os.path.join(f'{data_root}/entity2id_resp{version}.pkl')
    with open(reddit_test_resp_meta_pkl, "rb") as f:
        reddit_test_resp_id_name_table = pickle.load(f)

    reddit_test_resp_name2id = reddit_test_resp_id_name_table.set_index('title')['imdb_id'].to_dict()
    reddit_test_resp_id2name = {v: extract_movie_name(k) for k, v in reddit_test_resp_name2id.items()}
    relevant_indices = [col for col, imdb_id in raw_col2imdb_id.items() if imdb_id in reddit_test_resp_id2name]
    sim_mat = sim_mat[:, relevant_indices]
    col2imdb_id = {new_col: raw_col2imdb_id[old_col] for new_col, old_col in enumerate(relevant_indices)}
    imdb_id2col = {imdb_id: col for col, imdb_id in col2imdb_id.items()}
    
    catalog_imdb_ids = set(reddit_test_resp_id_name_table["imdb_id"]).intersection(set(raw_imdb_id2col.keys()))
    len(catalog_imdb_ids) / len(set(reddit_test_resp_id_name_table["imdb_id"]))
    
    ### get the title information
    old_meta_json = os.path.join(f'{data_root}/entity2id.json')
    old_name2id = json.load(open(old_meta_json))
    old_id2name = {v: extract_movie_name(k) for k, v in old_name2id.items()}
    
    database_root = "data/imdb_data"
    name_id_table = pd.read_csv(os.path.join(database_root, "imdb_titlenames_new.csv"))
    importance_table = pd.read_csv(os.path.join(database_root, "imdb_title_importance.csv"))
    importance_table = importance_table[importance_table["importance_tier"]<tier]
    name_id_table = pd.merge(name_id_table, importance_table, on='imdb_id', how='inner')
    name_id_table = name_id_table.sort_values(by='importance_rank')
    unique_titles = name_id_table.drop_duplicates(subset='title_name', keep='first')
    name2id = unique_titles.set_index('title_name')['imdb_id'].to_dict()
    id2name = {v: extract_movie_name(k) for k, v in name2id.items()}
    id2name.update(old_id2name)

    return sim_mat, catalog_imdb_ids, imdb_id2row, imdb_id2col, col2imdb_id, id2name
    

def pre_process(test_data, catalog_imdb_ids):
    '''
        This function pre-processes the data
    '''
    test_data = [item for item in test_data if item["old"]["is_user"] == 0]
    test_data = [item for item in test_data if all(iid in catalog_imdb_ids for iid in item["clean_resp_imdb_ids"])]

    # create a dictionary to keep track of seen turn_id
    seen_turn_ids = {}
    unique_test_data = []

    for item in test_data:
        turn_id = item["turn_id"]
        if turn_id not in seen_turn_ids:
            seen_turn_ids[turn_id] = True
            unique_test_data.append(item)

    # unique_test_data now contains items with unique turn_id
    test_data_with_rec = unique_test_data 
    return test_data_with_rec

    
def context_aware_retrieval(test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name):
    '''
        This module retrieves the collaborative knowledge
        and reflects on its contextual relevancy
    '''

    print(f"-----Context-aware Reflection-----")
    for K in K_list:
        EXSTING = {}
        threads, results = [], []
        
        for i, item in enumerate(tqdm(test_data_with_rec, 
                                    total=len(test_data_with_rec), 
                                    desc=f"reflecting on the retrieved titles - {K} raw retrieval...")):        
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
            
            flattened_triples = [
                (iid, title, attitude)
                for iids, titles, attitudes in zip(item["clean_context_imdb_ids"], item["clean_context_titles"], item["clean_context_attitudes"])
                for iid, title, attitude in zip(iids, titles, attitudes)
            ]
            
            ### filter titles with non-negative attitudes and ensure uniqueness
            context_ids = list({
                iid for iid, title, attitude in flattened_triples if attitude in {"0", "1", "2"}
            })
            context_titles = list({
                title for iid, title, attitude in flattened_triples if attitude in {"0", "1", "2"}
            })
        
            ### collaborative filtering augmented retrieval
            retrieved_titles, _ = cf_retrieve(context_ids, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name, K)
            retrieved_titles = ", ".join([f"{i+1}. {title}" for i, title in enumerate(retrieved_titles)])
        
            input_text = {
                "context" : context,
                "retrieved_titles" : retrieved_titles
            }
            prompt = prompt_reflect_titles
                        
            execute_thread = threading.Thread(
                target=get_response,
                args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
            )
            
            time.sleep(0.02)
            threads.append(execute_thread)
            execute_thread.start()
            if len(threads) == n_threads:
                for execute_thread in threads:
                    execute_thread.join()
        
                for res in results:
                    index = res["index"]
                    test_data_with_rec[index][f"reflect_retrieval_from_llm_{K}"] = res
                    
                threads = []
                results = []
                time.sleep(0)
        
        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()
        
        for res in results:
            index = res["index"]
            test_data_with_rec[index][f"reflect_retrieval_from_llm_{K}"] = res

    for K in K_list:
        test_data_with_rec = [process_retrieval_reflect_raw(item, K) for item in test_data_with_rec]
        errors = [item[0] for item in test_data_with_rec if item[0]]
        test_data_with_rec = [item[1] for item in test_data_with_rec]
        print(f"# errors for {K}: {len(errors)}") ## should be all zeros


def recommend_zero_shot(test_data_with_rec):
    '''
        This module conducts the zero-shot recommendation
        with LLM as the baseline
    '''
    EXSTING = {}
    threads, results = [], []

    print(f"-----Zero-shot Recommendation-----")
    for i, item in enumerate(tqdm(test_data_with_rec, 
                                total=len(test_data_with_rec), 
                                desc="generating recommendations...")):        
        context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
        
        input_text = {
            "context" : context,
        }
        prompt = prompt_no_title
            
        execute_thread = threading.Thread(
            target=get_response,
            args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
        )

        time.sleep(0.025)
        threads.append(execute_thread)
        execute_thread.start()
        if len(threads) == n_threads:
            for execute_thread in threads:
                execute_thread.join()

            for res in results:
                index = res["index"]
                test_data_with_rec[index][f"rec_from_llm_0"] = res
                
            threads = []
            results = []
            time.sleep(0)

    if len(threads) > 0:
        for execute_thread in threads:
            execute_thread.join()

    for res in results:
        index = res["index"]
        test_data_with_rec[index][f"rec_from_llm_0"] = res
        
    ### number of available cores
    num_cores = int(multiprocessing.cpu_count() / 1.5)
    print(f"number of cores: {num_cores}")

    test_data_with_rec = [process_item_raw(item, 0) for item in test_data_with_rec]
        

def recommend_with_retrieval(test_data_with_rec):
    '''
        This module conducts the CRAG recommendation with
        retrieved items as extra collaborative information
    '''
    for K in K_list:
        ### print current loop info
        print(f"-----CF-Augmented Recommendation-----")
    
        EXSTING = {}
        threads, results = [], []
        
        for i, item in enumerate(tqdm(test_data_with_rec, 
                                    total=len(test_data_with_rec), 
                                    desc=f"generating recommendations - {K} raw retrieval...")):        
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
        
            retrieved_titles = item[f"retrieval_after_reflect_{K}"]

            ### use retrieved titles after self-reflection
            if retrieved_titles:
                retrieved_titles = ", ".join([f"{i+1}. {title}" for i, title in enumerate(retrieved_titles)])
                input_text = {
                    "context" : context,
                    "retrieved_titles" : retrieved_titles
                }
                prompt = prompt_with_retrieved_titles

            ### deteriorate into zero-shot recommendation
            else:
                input_text = {
                    "context" : context,
                }
                prompt = prompt_no_title                         
                
            execute_thread = threading.Thread(
                target=get_response,
                args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
            )

            time.sleep(0.025)
            threads.append(execute_thread)
            execute_thread.start()
            if len(threads) == n_threads:
                for execute_thread in threads:
                    execute_thread.join()
        
                for res in results:
                    index = res["index"]
                    test_data_with_rec[index][f"rec_from_llm_{K}"] = res
                    
                threads = []
                results = []
                time.sleep(0)
        
        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()
        
        for res in results:
            index = res["index"]
            test_data_with_rec[index][f"rec_from_llm_{K}"] = res
            

    ### number of available cores
    num_cores = int(multiprocessing.cpu_count() / 1.5)
    print(f"number of cores: {num_cores}")

    for K in K_list:
        test_data_with_rec = [process_item_raw(item, K) for item in test_data_with_rec]

    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, "test_with_retrieval.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(test_data_with_rec, f)

    print(f"results saved to: {save_file}!")
        

def reflect_and_rerank(test_data_with_rec):
    '''
        This module reflects upon the final recommendation list
        and rerank them based on the relevance score.
    '''
    K_list = list(range(5, 40, 5))
    if "rec_from_llm_0" in test_data_with_rec[0] and 0 not in K_list:
        K_list = [0] + K_list

    for K in K_list:
        print(f"-----Reflect and Rerank-----")
        
        EXSTING = {}
        threads, results = [], []
        
        for i, item in enumerate(tqdm(test_data_with_rec, 
                                    total=len(test_data_with_rec), 
                                    desc=f"reflect and rerank the recommended titles - {K} raw retrieval...")):        
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])

            rec_list_raw = item[f"rec_list_raw_{K}"]
            rec_titles = ", ".join([f"{i+1}. {title}" for i, title in enumerate(rec_list_raw)])
        
            input_text = {
                "context" : context,
                "rec_titles" : rec_titles
            }
            prompt = prompt_reflect_rec_titles
                        
            execute_thread = threading.Thread(
                target=get_response,
                args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
            )
            
            time.sleep(0.02)
            threads.append(execute_thread)
            execute_thread.start()
            if len(threads) == n_threads:
                for execute_thread in threads:
                    execute_thread.join()
        
                for res in results:
                    index = res["index"]
                    test_data_with_rec[index][f"reflect_rec_from_llm_{K}"] = res
                    
                threads = []
                results = []
                time.sleep(0)
        
        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()
        
        for res in results:
            index = res["index"]
            test_data_with_rec[index][f"reflect_rec_from_llm_{K}"] = res
            

    for K in K_list:
        test_data_with_rec = [process_rec_reflect_raw(item, K) for item in test_data_with_rec]
        errors = [item[0] for item in test_data_with_rec if item[0]]
        test_data_with_rec = [item[1] for item in test_data_with_rec]
        print(f"# errors for {K}: {len(errors)}")

    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, "test_with_reflect_and_rerank.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(test_data_with_rec, f)

    print(f"results saved to: {save_file}!")
        
        
def post_processing(test_data_with_rec):
    ### obtain the groundtruth filtering the resp titles
    for item in test_data_with_rec:
        clean_resp_titles = item["clean_resp_titles"]
        clean_resp_attitude = item["clean_resp_attitude"]
        clean_context_titles = item["clean_context_titles"]
        
        groundtruth = []
        
        for title, attitude in zip(clean_resp_titles, clean_resp_attitude):
            if attitude not in ("-2", "-1") and not any(title in context for context in clean_context_titles):
                groundtruth.append(title)
        
        item["groundtruth"] = groundtruth
    test_data_with_rec = [item for item in test_data_with_rec if (item["old"]["is_user"] == 0) and (item["groundtruth"])]
    
    return test_data_with_rec
        

def evaluate_no_rerank(test_data_with_rec_filtered):
    '''
        Evaluate the performance when NO reflect and rerank
        is imposed on the final recommendation list
    '''
    avg_metrics_filtered = {}
    metrics_filtered = {}
    
    for K in K_list:
        print(f"Processing {K} retrievals")

        errors = set()
        results = {k:[] for k in k_list}
        
        for k in k_list:
            print(f"Processing top {k}")
            for i, item in enumerate(tqdm(test_data_with_rec_filtered, 
                                     total=len(test_data_with_rec_filtered), 
                                     desc="Evaluating via direct match...")):
                try:
                    results[k].append(evaluate_direct_match(item, k, K, gt_field="groundtruth"))
                except:
                    errors.add(i)
        
        recalls = {k:[res[0] for res in results[k]] for k in k_list}
        ndcgs = {k:[res[1] for res in results[k]] for k in k_list}

        metrics_filtered[K] = (recalls, ndcgs)
        
        avg_recalls_filtered = {k:np.mean(recalls[k]) for k in k_list}
        avg_ndcgs_filtered = {k:np.mean(ndcgs[k]) for k in k_list}
        avg_metrics_filtered[K] = (avg_recalls_filtered, avg_ndcgs_filtered)
        
        print(f"number of errors: {len(errors)}")

    metrics["No Reflect and Rerank"] = metrics_filtered
    avg_metrics["No Reflect and Rerank"] = avg_metrics_filtered
    
    for K in K_list:
        print(f"{K}-retrieval results:")
        avg_recalls_filtered, avg_ndcgs_filtered = avg_metrics_filtered[K]    
        print("Results on the filtered dataset:")
        print(f"top-k recalls: {avg_recalls_filtered}")
        print(f"top-k ndcgs: {avg_ndcgs_filtered}\n")
    

def evaluate_with_rerank(test_data_with_rec_filtered):
    '''
        Evaluate the performance when reflect and rerank
        is imposed on the final recommendation list
    '''
    metrics_reflect_rerank = {}
    avg_metrics_reflect_rerank = {}

    for K in K_list:
        print(f"Processing {K} retrievals")

        errors = set()
        results = {k:[] for k in k_list}
        
        for k in k_list:
            print(f"Processing top {k}")
            for i, item in enumerate(tqdm(test_data_with_rec_filtered, 
                                     total=len(test_data_with_rec_filtered), 
                                     desc="Evaluating via direct match...")):
                try:
                    results[k].append(evaluate_direct_match_reflect_rerank(item, k, K, gt_field="groundtruth"))
                except:
                    errors.add(i)
        
        recalls = {k:[res[0] for res in results[k]] for k in k_list}
        ndcgs = {k:[res[1] for res in results[k]] for k in k_list}

        metrics_reflect_rerank[K] = (recalls, ndcgs)

        avg_recalls_reflect_rerank = {k:np.mean(recalls[k]) for k in k_list}
        avg_ndcgs_reflect_rerank = {k:np.mean(ndcgs[k]) for k in k_list}
        avg_metrics_reflect_rerank[K]= (avg_recalls_reflect_rerank, avg_ndcgs_reflect_rerank)
        
        print(f"number of errors: {len(errors)}")

    metrics["With Reflect and Rerank"] = metrics_reflect_rerank
    avg_metrics["With Reflect and Rerank"] = avg_metrics_reflect_rerank

    for K in K_list:
        print(f"{K}-retrieval results:")
        avg_recalls_reflect_rerank, avg_ndcgs_reflect_rerank = avg_metrics_reflect_rerank[K]       
        print("Results on the filtered dataset:")
        print(f"top-K recalls: {avg_recalls_reflect_rerank}")
        print(f"top-K ndcgs: {avg_ndcgs_reflect_rerank}\n")
        

def simple_plot(save_file):
    '''
        A simple plot function to visualize the results
    '''
    num_methods = len(avg_metrics)
    fig, axs = plt.subplots(num_methods, 2, figsize=(10, 3 * num_methods), facecolor='none', edgecolor='none')

    if num_methods == 1:
        axs = [axs]

    for method_idx, (method, results) in enumerate(avg_metrics.items()):
        ### prepare data for plotting recalls and ndcgs
        recalls = {k: [] for k in k_list}
        ndcgs = {k: [] for k in k_list}

        ### collecting recall and ndcg values for each k
        for k_setting, (recall_dict, ndcg_dict) in results.items():
            for k in k_list:
                recalls[k].append(recall_dict[k])
                ndcgs[k].append(ndcg_dict[k])

        ### plotting recall results
        ax = axs[method_idx][0]
        num_K = len(results.keys())
        x = np.arange(len(k_list))  # label locations
        width = 0.8 / num_K  # the width of the bars

        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, num_K)

        for idx, k_setting in enumerate(results.keys()):
            rects = ax.bar(x + offsets[idx], [recalls[k][idx] for k in k_list], width, label=k_setting)

        ax.set_xlabel('Top k')
        ax.set_ylabel('Recall')
        ax.set_title(f'Recall at different k for {method}')
        ax.set_xticks(x)
        ax.set_xticklabels(k_list)

        ### plotting NDCG results
        ax = axs[method_idx][1]
        x = np.arange(len(k_list))  # label locations
        width = 0.8 / num_K  # the width of the bars

        for idx, k_setting in enumerate(results.keys()):
            rects = ax.bar(x + offsets[idx], [ndcgs[k][idx] for k in k_list], width, label=k_setting)

        ax.set_xlabel('Top k')
        ax.set_ylabel('NDCG')
        ax.set_title(f'NDCG at different k for {method}')
        ax.set_xticks(x)
        ax.set_xticklabels(k_list)

    ### collect handles for legend
    handles, labels = axs[0][0].get_legend_handles_labels()

    ### adding a common legend below all subplots
    fig.legend(handles, labels, loc='lower center', ncol=4, title="Problem Setting", bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    
    ### save the figure to a file
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fig.savefig(os.path.join(save_dir, "result.jpg"), format='jpg', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()
    

def main():
    ### load the collaborative model
    sim_mat, catalog_imdb_ids, imdb_id2row, imdb_id2col, col2imdb_id, id2name = load_and_process_cf_model()
    
    ### load the raw dataset
    with open(from_pkl, "rb") as f:
        test_data = pickle.load(f)
    
    ### pre-process the dataset
    test_data_with_rec = pre_process(test_data, catalog_imdb_ids)
    
    ### the main modules for CRAG
    context_aware_retrieval(test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name)
    recommend_zero_shot(test_data_with_rec)
    recommend_with_retrieval(test_data_with_rec)
    reflect_and_rerank(test_data_with_rec)

    # save_dir = "results"
    # save_file = os.path.join(save_dir, "test_with_reflect_and_rerank.pkl")
    # with open(save_file, "rb") as f:
    #     test_data_with_rec = pickle.load(f)   
    
    ### post-process the dataset
    test_data_with_rec = post_processing(test_data_with_rec)
    
    ### evaluate before and after the rerank
    avg_metric_no_rerank = evaluate_no_rerank(test_data_with_rec)
    avg_metric_with_rerank = evaluate_with_rerank(test_data_with_rec)
    
    ### plot the results
    simple_plot(avg_metric_no_rerank)


if __name__ == '__main__':
    main()