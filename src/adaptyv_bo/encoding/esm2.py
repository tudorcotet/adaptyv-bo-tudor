import transformers
from transformers import AutoModel, AutoTokenizer
import torch
from encoding.base import BaseEncoding 
from typing import List
import numpy as np

class ESM2Encoding(BaseEncoding):
    def __init__(self, model_name: str = "esm2_t36_3B_UR50D", pooling_method: str = "mean", sequence_list: List[str] = None, embedding_method: str = "pca", n_components: int = 2):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling_method = pooling_method
        self.sequence_list = sequence_list
        self.embedding_method = embedding_method
        self.n_components = n_components
        if self.embedding_method is not None:
            embedding_projection = EncodingEmbedder(self, self.embedding_method, self.n_components)

    def encode(self, sequences: List[str]) -> np.ndarray:

        # Tokenize the sequences
        encoded_sequences = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)

        # Get the hidden states from the model
        with torch.no_grad():
            outputs = self.model(**encoded_sequences)
            hidden_states = outputs.hidden_states[-1]

        if self.pooling_method == "mean":
            # Get the mean of the hidden states for each sequence
            sequence_embeddings = torch.mean(hidden_states[:, 1:-1, :], dim=1)
        elif self.pooling_method == "cls":
            # Get the CLS token's hidden state for each sequence
            sequence_embeddings = hidden_states[:, 0, :]
        elif self.pooling_method == "concat":
            # Get the CLS token's hidden state for each sequence
            sequence_embeddings = hidden_states[:, 1:-1, :].flatten(start_dim=1)
        elif self.pooling_method == "residues":
            if self.residue_list is not None:
                sequence_embeddings = hidden_states[:, 1:-1, :]
                sequence_embeddings = sequence_embeddings[:, self.residue_list, :]
            else:
                raise ValueError("Residue list is not provided")

        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")

        if self.embedding_method is not None:
            sequence_embeddings = embedding_projection.encode(sequence_embeddings.numpy())

        return sequence_embeddings

    def decode(self, encoded_sequences: np.ndarray) -> List[str]:
        return self.tokenizer.decode(encoded_sequences) 


class EncodingEmbedder(BaseEncoding):
    def __init__(self, encoding: BaseEncoding, method: str = "pca", n_components: int = 2):
        self.method = method
        self.encoding = encoding
        self.n_components = n_components

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        if self.method == "pca":
            pca = PCA(n_components=self.n_components)
            return pca.fit_transform(embeddings)
        elif self.method == "tsne":
            tsne = TSNE(n_components=self.n_components)
            return tsne.fit_transform(embeddings)
        elif self.method == "umap":
            umap = UMAP(n_components=self.n_components)
            return umap.fit_transform(embeddings)
        elif self.method == 'pacmap':
            pacmap = PaCMAP(n_components=self.n_components)
            return pacmap.fit_transform(embeddings)
        else:
            raise ValueError(f"Invalid embedding method: {self.method}")

    