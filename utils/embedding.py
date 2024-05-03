import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.mixture import GaussianMixture
from rank_bm25 import BM25Okapi
import numpy as np
import umap

class SBert:
    def __init__(self, model_name, max_length = 128):
        self.model = SentenceTransformer(model_name)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    
        
    def encode(self, text):
        with torch.no_grad():
            return self.model.encode(text)
    
    
    def cosine_similarity(self, text, text2):
        text = self.model.encode(text)
        text2 = self.model.encode(text2)
        if len(text.shape) == 1:
            text = np.expand_dims(text, 0)
        if len(text2.shape) == 1:
            text2 = np.expand_dims(text2, 0)
        
        text = text/np.expand_dims(np.sqrt(np.sum(text**2, axis=1)), 1)
        text2 = text2/np.expand_dims(np.sqrt(np.sum(text2**2, axis=1)), 1)
        
        similarity = np.dot(text, text2.T)
        return similarity
        
class ClusterRAG:
    def __init__(self, model_name,text = None, dim= None, max_cluster = 50, path = None, bm25 = True):
        self.sbert = SBert(model_name)
        if path:
            self.load_cluster(path)
        else:
            self.create_cluster(text, dim, max_cluster)
        
        self.is_bm25 = bm25
        if bm25:
            self.text = text
            tokenized_corpus = [sen.split() for sen in text]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
    def load_cluster(self, path):
        self.embed_text = np.load(path['embed_path'])
        self.embed_text_umap = np.load(path['embed_path'])
        self.embed_cluster = np.load(path['cluster_path'])
        self.lookup_sentence = np.embed_cluster.T
    
    def _umap(self, encode_text, n_neighbors = None):
        if n_neighbors is None:
            n_neighbors = int((len(encode_text) - 1) ** 0.5)
            
        self.umap = umap.UMAP(n_neighbors=n_neighbors,
                         n_components = self.dim,
                         metric = 'cosine')
        
        return self.umap.fit_transform(encode_text)
    
    def create_cluster(self, text, dim, max_cluster = 50):
        self.embed_text = self.sbert.encode(text)
        self.dim = dim

        embed_text = self._umap(self.embed_text)
        self.embed_text_umap = embed_text
        
        max_cluster = min(max_cluster, len(embed_text))
        bics = []
        n_clusters = np.arange(1, max_cluster)
        
        for i in range(1, max_cluster):
            gmm = GaussianMixture(n_components = i)
            gmm.fit(embed_text)
            bics.append(gmm.bic(embed_text))
            
        self.n_clusters = n_clusters[np.argmin(bics)]
        
        self.gmm = GaussianMixture(n_components = self.n_clusters)
        self.gmm.fit(embed_text)
        self.embed_cluster = self.gmm.predict_proba(embed_text)
        self.embed_cluster = np.array([np.where(line > 0.3)[0] for line in self.embed_cluster]).astype(bool)
        self.lookup_sentence = self.embed_cluster.T
        
        
    def _cosine_similarity(self, embed_text, mask_cluster, text, topk = 3):

        cosine = np.dot(embed_text, self.embed_text_umap[mask_cluster].T)/ \
                    (np.expand_dims(np.linalg.norm(embed_text, axis=1), 1) 
                     * np.expand_dims(np.linalg.norm(self.embed_text_umap[mask_cluster], axis = 1)), 1)
                    
        cosine = cosine.squeeze()
        
        if self.is_bm25:
            tokenized_query = text.split()
            cosine = cosine * self.bm25.get_scores(tokenized_query)
        
        return cosine.argsort()[-topk:][::-1]
    
    
    def lookup(self, text, topk = 3):
        embed_text = self.sbert.encode(text)
        if len(embed_text.shape) == 1:
            embed_text = np.expand_dims(embed_text, 0)
            
        umap_text = self.umap.transform(embed_text)
        cluster = self.gmm.predict(umap_text)
        
        mask_cluster = self.lookup_sentence[cluster]
        
        return self._cosine_similarity(embed_text, mask_cluster, text, topk=topk)
    
    
        
if __name__ == "__main__":
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    sbert = SBert(model_name)
    sen1 = 'I love you'
    sen2 = 'I like you'
    print(sbert.cosine_similarity([sen1, sen2],[sen1, sen2]))
    