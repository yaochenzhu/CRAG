import os
import re
import json
import regex
import pickle

import datetime
from time import sleep

import hashlib
import numpy as np

from collections import defaultdict
from editdistance import eval as distance

from rank_bm25 import BM25Okapi

def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)


def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()


def del_numbering(text):
    pattern = r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?"
    return re.sub(pattern, "", text)


def remove_quotes(s):
    if (s.startswith('"') and s.endswith('"')) \
        or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def is_in(text, items, threshold):
    for i in items:
        if (distance(i.lower(), text.lower()) <= threshold):
            return True
    return False


def nearest(text, items):
    """ given the raw text name and all candidates, 
        return {movie_name:, min_edit_distance: , nearest_movie: }
    """
    # calculate the edit distance
    items = list(set(items))
    dists = [distance(text.lower(), i.lower()) for i in items]
    # find the nearest movie
    nearest_idx = np.argmin(dists)
    nearest_movie = items[nearest_idx]
    return {
        'movie_name': text, 
        'min_edit_distance': dists[nearest_idx], 
        'nearest_movie': nearest_movie
    }


def nearest_thres(text, items, thres):
    """ given the raw text name and all candidates, 
        return {movie_name:, min_edit_distance: , nearest_movie: }
    """
    # calculate the edit distance
    items = list(set(items))
    dists = [distance(text.lower(), i.lower()) for i in items]
    # find the nearest movie
    nearest_idx = np.argmin(dists)
    nearest_movie = items[nearest_idx]

    if dists[nearest_idx] <= thres:
        return {
            'movie_name': text, 
            'min_edit_distance': dists[nearest_idx], 
            'nearest_movie': nearest_movie
        }
    else:
        return None


def extract_movie_name(text):
    text = text.split('/')[-1]
    text = text.replace('_', ' ').replace('-', ' ').replace('>', ' ')
    return del_space(del_parentheses(text))


def process_retrieval_reflect_raw(item, K):
    raw_retrieval_reflect = item[f'reflect_retrieval_from_llm_{K}']['resp']['choices'][0]['message']['content']
    raw_retrieval_reflect = re.sub(r'\n+', '\n', raw_retrieval_reflect)
    error = False
    try:
        raw_retrieval_reflect_list = [reflect.strip().split("####") for reflect in raw_retrieval_reflect.split('\n')]
        raw_retrieval_reflect_list = [item for item in raw_retrieval_reflect_list if len(item) == 2]
        raw_retrieval_reflect_list = [title for title, judgment in raw_retrieval_reflect_list if judgment == "1"]
        item[f"retrieval_after_reflect_{K}"] = [del_numbering(del_space(del_parentheses(i.strip()))) for i in raw_retrieval_reflect_list]
    except:
        item[f"retrieval_after_reflect_{K}"] = []
        error = True
    return error, item


def process_item_raw(item, K):
    raw_rec = item[f'rec_from_llm_{K}']['resp']['choices'][0]['message']['content']
    ### Remove extra lines
    raw_rec = re.sub(r'\n+', '\n', raw_rec)
    ### Standardize movie names
    raw_rec_list = [del_numbering(del_space(del_parentheses(remove_quotes(i).strip()))) for i in raw_rec.split('\n')]
    ### Remove the remaining quotes
    raw_rec_list = [remove_quotes(name) for name in raw_rec_list]
    item[f"rec_list_raw_{K}"] = raw_rec_list
    return item


def rating_key(movie):
    rating = movie[1]
    # Create a mapping for the ratings
    rating_map = {"2": 5, "1": 4, "0": 3, "-1": 2, "-2": 1}
    # Return the mapped value or 0 for other strings
    return rating_map.get(rating, 0)


def process_rec_reflect_raw(item, K):
    raw_rec_reflect = item[f'reflect_rec_from_llm_{K}']['resp']['choices'][0]['message']['content']
    raw_rec_reflect = re.sub(r'\n+', '\n', raw_rec_reflect)
    try:
        raw_rec_reflect_list = [reflect.strip().split("####") for reflect in raw_rec_reflect.split('\n')]
        raw_rec_reflect_list = [item for item in raw_rec_reflect_list if len(item) == 2]
        # Sort the list using the custom key function, maintaining original order for same ratings
        raw_rec_reflect_list = sorted(raw_rec_reflect_list, key=rating_key, reverse=True)
        # Extract the movie names from the sorted list
        raw_rec_reflect_list = [movie[0] for movie in raw_rec_reflect_list]     
        item[f"rec_after_reflect_{K}"] = [del_numbering(del_space(del_parentheses(i.strip()))) for i in raw_rec_reflect_list]
        error = False
    except:
        item[f"rec_after_reflect_{K}"] = item[f"rec_list_raw_{K}"]
        error = True
    return error, item


class TitleInfo():
    def __init__(self,title_info):
        self.title_info=title_info
        self.title_names = title_info[ "title_name"].to_numpy()
        self.title_importance = title_info[ "importance_tier"].to_numpy()
        self.title_imdb_id = title_info[ "imdb_id"].to_numpy()
        self.tokenized_title_names = [  self.preprocess_phrase(doc).split(" ") for doc in self.title_names  ]
        self.bm25 = BM25Okapi(self.tokenized_title_names)

        self.imdb_id2ix_dict={}
        for ix,iid in  enumerate(self.title_imdb_id):
            self.imdb_id2ix_dict[iid]=ix

    def imdb_id_2_title_name(self, iid):
        if iid in self.imdb_id2ix_dict.keys(): 
            return self.title_names[  self.imdb_id2ix_dict[iid] ]
        return ""
        
    def imdb_id_2_title_importance(self, iid):
        if iid in self.imdb_id2ix_dict.keys(): 
            return self.title_importance[  self.imdb_id2ix_dict[iid] ]
        return -1        
    
    def preprocess_phrase(self,phrase):
        co1=re.sub(r"\W", " ", phrase)
        co2=re.sub(r"\s+", " ", co1)
        co3=co2.strip().lower()
        return co3

    def find_best_title_matches(self, query):
        preprocessed_query= self.preprocess_phrase(query)
        tokenized_query = preprocessed_query.split(" ")
        
        ## find good matches using BM25
        scores = self.bm25.get_scores(tokenized_query)
        kk=np.argsort(-scores)
        kk=kk[scores[kk]>0.0]
        ii=kk[:100]
        #### most words are matching
        ii_matches=[0]*len(ii)

        if not ii_matches:
            return [0, []]
            
        for c,i in enumerate(ii):
            for tt in tokenized_query:
                if tt in self.tokenized_title_names[i]: ii_matches[c]+=1
        jj=ii[np.array(ii_matches) == max(ii_matches)]
        query_match_fraction = max(ii_matches) /1.00/len(tokenized_query )
        #### re-sort best matches by title-importance
        kk = np.argsort(self.title_importance[jj])
        jj=jj[kk]
        ### put exact matches of title-names to the front
        exact=[]
        for j in jj:
            if  preprocessed_query ==  " ".join(self.tokenized_title_names[j]):
                exact.append(j)
        ii=[]
        for j in jj:
            if j not in exact:
                ii.append(j)
        jj= exact+ii
        return [query_match_fraction,  self.title_imdb_id[jj]]
