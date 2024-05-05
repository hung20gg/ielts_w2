import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.mixture import GaussianMixture
from rank_bm25 import BM25Okapi
import numpy as np
import umap
import pandas as pd
from tqdm import tqdm

class SBert:
    def __init__(self, model_name, dtype='auto', max_length = 128):
        self.model = SentenceTransformer(model_name)
        
        if dtype == 'bfloat16':
            self.model = self.model.to(torch.bfloat16)
        else:
            self.model = self.model.half()
            
        self.model = self.model.eval()
        
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    
        
    def encode(self, text):
        with torch.no_grad():
            embed =  self.model.encode(text, convert_to_tensor=True, show_progress_bar=False, device=self.device)
            embed = embed.cpu().numpy()
        return embed
    
    
    def cosine_similarity(self, text, text2):
        text = self.encode(text)
        text2 = self.encode(text2)
        if len(text.shape) == 1:
            text = np.expand_dims(text, 0)
        if len(text2.shape) == 1:
            text2 = np.expand_dims(text2, 0)
        
        text = text/np.expand_dims(np.sqrt(np.sum(text**2, axis=1)), 1)
        text2 = text2/np.expand_dims(np.sqrt(np.sum(text2**2, axis=1)), 1)
        
        similarity = np.dot(text, text2.T)
        return similarity
        
class ClusterRAG:
    def __init__(self, model_name,
                 dim= 32, max_cluster = 50, 
                 path = None, bm25 = True,
                 is_cluster = True,
                 dtype = 'auto',
                 df_path = '../data/lookup_essay.csv'):
        self.sbert = SBert(model_name, dtype)
        self.is_cluster = is_cluster
        if path:
            self.load_cluster(path)
        else:
            self.load_essay(df_path)
            self.embed_topics = self.sbert.encode(self.topic)
            if is_cluster:
                self.create_cluster(dim, max_cluster)
        
        # BM25 for more accurate similarity
        self.is_bm25 = bm25
        if bm25:
            print('BM25 is enabled')
            tokenized_corpus = [sen.split() for sen in self.topic]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
    def load_cluster(self, path):
        """Load the cluster from the path

        """
        self.df = pd.read_csv(path['df']) 
        self.topic = self.df['topic'].unique()
        self.embed_topics = np.load(path['embed_path'])
        self.embed_cluster = np.load(path['cluster_path'])
        self.lookup_sentence = np.embed_cluster.T
    
    def _umap(self, encode_text, n_neighbors = None):
        if n_neighbors is None:
            n_neighbors = int((len(encode_text) - 1) ** 0.5)
            
        self.umap = umap.UMAP(n_neighbors=n_neighbors,
                         n_components = self.dim,
                         metric = 'cosine')
        
        return self.umap.fit_transform(encode_text)
    
    def load_essay(self, df_path):
        print('Loading essay')
        self.df = pd.read_csv(df_path) 
        self.topic = self.df['topic'].unique()
    
    def _get_essay_and_comment(self, topic_id, essay, topk = 1):
        topic_index = []
        essays = []
        comments = []
        
        for id in topic_id:
            sub_df = self.df[self.df['topic_idx'] == id][:max(1, 2*topk//2)]
            for es, co in zip(sub_df['text'].values,sub_df['comment'].values): 
                topic_index.append(id)
                essays.append(es)
                comments.append(co)
                
            
        
        topic_index = np.array(topic_index)
        essays = np.array(essays)
        comments = np.array(comments)
        
        cosine = self.sbert.cosine_similarity(essay, essays)
        
        get = np.argsort(cosine)[:, -topk:][:, ::-1]
        
        return self.topic[topic_index[get]], essays[get], comments[get]
        
    def encode(self, text, force=False):
        text = self.sbert.encode(text)
        if len(text.shape) == 1:
            text = np.expand_dims(text, 0)
        if not self.is_cluster or force:
            return text

        return self.umap.transform(text)
    
    def create_cluster(self, dim, max_cluster = 50):
        self.dim = dim
        if self.is_cluster:

            embed_topics = self._umap(self.embed_topics)
        else:
            embed_topics = self.embed_topics
        
        max_cluster = min(max_cluster, len(embed_topics))
        bics = []
        n_clusters = np.arange(1, max_cluster)
        
        # for i in tqdm(range(1, max_cluster), desc='Clustering'):
        #     gmm = GaussianMixture(n_components = i)
        #     gmm.fit(self.embed_topics)
        #     bics.append(gmm.bic(self.embed_topics))
            
        # self.n_clusters = n_clusters[np.argmin(bics)]
        
        
        
        # For testing purpose, cluster is 14
        
        self.n_clusters = 12
        
        print(f'Number of clusters: {self.n_clusters}')
        self.cluster = GaussianMixture(n_components = self.n_clusters)
        self.cluster.fit(embed_topics)
        self.embed_cluster = self.cluster.predict_proba(embed_topics)

        self.embed_cluster = np.array([np.where(line > 0.1, 1, 0) for line in self.embed_cluster]).astype(bool)
        print(self.embed_cluster.shape)
        self.lookup_sentence = self.embed_cluster.T
        
        print(np.sum(self.lookup_sentence, axis = 1))
        
        
    def _cosine_similarity(self, text, mask_cluster = np.array([]), topk = 10):
        if mask_cluster.any():
            embed_topics = self.embed_topics[mask_cluster]
        else:
            embed_topics = self.embed_topics
        embed_text = self.encode(text, force=True)
        
        text_norm = np.linalg.norm(embed_text, axis=1)
        topics_norm = np.linalg.norm(embed_topics, axis = 1)
        
        cosine = np.dot(embed_text, embed_topics.T) / (np.expand_dims(text_norm, 1) * np.expand_dims(topics_norm, 0))
                    
        
        if self.is_bm25:
            tokenized_query = text.split()
            cosine = cosine * (1 + self.bm25.get_scores(tokenized_query))
        
        return cosine.argsort(axis = -1)[:, ::-1][:, :topk]
    
    
    def lookup(self, text, topk = 10):
        
        
        mask_cluster = np.array([])
        if self.is_cluster:
            embed_text = self.encode(text)
            cluster = self.cluster.predict(embed_text)[0]

            mask_cluster = self.lookup_sentence[cluster]
        return self._cosine_similarity(text, mask_cluster, topk=topk)
    
    def retrieve(self, topic, essay, topk = 10):
        id = self.lookup(topic, topk).squeeze()
        similar_topic, similar_essay, comment = self._get_essay_and_comment(id, essay, topk)
        return similar_topic[0], similar_essay[0], comment[0]
        

        
if __name__ == "__main__":
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    RAG = ClusterRAG(model_name, dtype='float16', is_cluster = False)
    # print(dir(umap))
    
    example_dir = '../sample/lanvy/1_4.5-5.5'
    with open(f'{example_dir}/question.txt', 'r') as f:
        topic = f.read()
    with open(f'{example_dir}/answer.txt', 'r') as f:
        essay = f.read()
        
    # index = RAG.lookup(topic)
    # print('index: ', index)
    # for text in RAG.topic[index[0]]:
    #     print(text)    

    topic, essay, comment = RAG.retrieve(topic, essay, topk = 3)
    for t, e, c in zip(topic, essay, comment):
        print(t)
        print(e)
        print(c)
        print('=====================')
    #     print(topic)
    #     print(essay)
    #     print(comment)
    #     print('=====================')