# CRAG: Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems

The code is associated with the following paper [[pdf]](https://arxiv.org/abs/2502.14137):

>Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems    
>**Yaochen Zhu**, Chao Wan, Harald Steck, Dawen Liang, Yesu Feng, Nathan Kallus, Jundong Li    
>The ACM Web Conference (WWW) 2025.

which is a joint research from the University of Virginia [VAST LAB](https://jundongli.github.io/), Cornell University, and Netflix Inc.


## 1. Reddit-v2 Dataset

We are currently organizing the refined Reddit-v2 dataset. However, if you wish to reproduce the experiments, you can download the unorganized test set (with lots of intermediate results from GPT-4o and bi-level matching step), the learned EASE model [[here]](https://drive.google.com/file/d/1h_CIyLP0_pIF2kfVBS6jJOxy6Ux6TWkm/view?usp=drive_link), and additional movie-related metadata [[here]](https://drive.google.com/file/d/1PLxHu-claqgI_yPm1zQHG97mHob7xmUz/view?usp=sharing). After downloading, simply unzip the files and place them in the data directory.

We will release the Reddit-v2 training set as soon as we finish the data organization. 

## !! Update

We decide to release the unorganized Reddit-v2 dataset first. Even though it contains lots of intermediate results from GPT-4o and bi-level matching, we believe that these results could be useful to many future researchers to avoid substantial API cost to extract the movie names and match them to the database.

The link to the Reddit-v2 dataset is [[here]](https://drive.google.com/file/d/19rEYwEOlVFs04-PYQFzhfbgxLu42-mvl/view?usp=sharing)

Specifically, the EASE model is trained by combining all the training data in both large, small, and CIKM subdataset, and excluding the test data, which is a subset of the CIKM test set.

## 2. How to Run the Code

To execute the code, please follow these steps:

**a)** Set ```external = True``` in both evaluate.py and libs/model.py as Netflix has its own method for accessing OpenAI models.

**b)** Configure your ```OPENAI_API_KEY``` as a system environment variable.

**c)** Run the following command:  
```python evaluate.py```    
The numerical results, along with a simple plot resembling Fig. 2, will be saved in the result folder.


## 3. Caveats

**a)** Please be aware that due to the bandwidth that we had during my internship, we have set the sleep time between threads of OpenAI API calls to 0.02 seconds. For individual users, this may exceed the limit. You can adjust this number to better fit your bandwidth. 

**b)** Additionally, we did not encounter any failed requests during our experiments, so no post-fixing was implemented. If you anticipate API failures (or printed #error for processing LLM's outputs to be non-zero), you may need to modify the code to handle such scenarios.


## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```
@inproceedings{zhu2025collaborative,
  title={Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems},
  author={Zhu, Yaochen and Wan, Chao and Steck, Harald and Liang, Dawen and Yesu, Feng and Kallus, Nathan and Li, Jundong},
  booktitle={Proceedings of the ACM Web Conference},
  year={2025}
}
```
**Thanks for your interest in our work!**
