import time
import openai
from copy import deepcopy

import numpy as np

external = False
if not external:
    import nflx_copilot as ncp
    openai = ncp
    ncp.project_id = "reddit"

def cf_retrieve(imdb_ids, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name, K):
    # Initialize an array to sum the rows corresponding to the given imdb_ids
    total_sim = np.zeros(sim_mat.shape[1])
    
    # Sum the rows corresponding to the imdb_ids
    for imdb_id in imdb_ids:
        if imdb_id in imdb_id2row:
            row_index = imdb_id2row[imdb_id]
            total_sim += sim_mat[row_index]
        
    # Set the position that corresponds to the specific imdb_id to zero to prevent self-recommendation
    for imdb_id in imdb_ids:
        if imdb_id in imdb_id2col:
            col_index = imdb_id2col[imdb_id]
            total_sim[col_index] = 0

    # Get indices of sorted similarities in descending order
    sorted_indices = np.argsort(-total_sim)
    
    # Collect top K or non-zero items
    top_k_names = []
    top_k_scores = []
    count = 0
    
    for index in sorted_indices:
        if total_sim[index] > 0:
            imdb_id = col2imdb_id[index]
            top_k_names.append(id2name[imdb_id])
            top_k_scores.append(total_sim[index])
            count += 1
        if count >= K:
            break
    return top_k_names, top_k_scores


def get_response(index, text, prompt, model, temperature, max_tokens, results, EXSTING):
    try:
        content = prompt.format(**text)

        if content in EXSTING:
            result = deepcopy(EXSTING[content])
            result['index'] = index
            print("Found in EXSTING!")
        else:
            resp = openai.ChatCompletion.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": content},
                ]
            )
            result = {'index': index, 'prompt': content, 'resp': resp}
            EXSTING[content] = result

        results.append(result)

    except Exception as e:

        if e == KeyboardInterrupt:
            raise e
        print(e)
        time.sleep(2)
        results.append({'index': index, 'prompt': content,
                       'resp': "API Failed"})