import torch
from speechbrain.inference.speaker import EncoderClassifier
from .host_similarity_mean import HostSimilarityMean
import numpy as np
from pyannote.audio.pipelines.clustering import BaseClustering, AgglomerativeClustering

torch.device('cuda')


class CustomCluster(BaseClustering):
    def __init__(self, classifier: EncoderClassifier):
        super().__init__()
        self.classifier = classifier

    def cluster(
        self,
        embeddings: np.ndarray,
        # these args is dummy
        min_clusters: int = 1,
        max_clusters: int = 2,
        num_clusters: int = 2
    ):  
        return embeddings.argmax(axis = -1)
