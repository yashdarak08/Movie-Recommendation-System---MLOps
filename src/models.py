import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MatrixFactorizationModel(nn.Module):
    """
    Matrix Factorization model for collaborative filtering.
    
    This model represents users and items as embeddings and computes
    the dot product between them to predict ratings.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        """
        Initialize matrix factorization model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Dimension of embedding vectors
        """
        super(MatrixFactorizationModel, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings with small random values
        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        self.item_embedding.weight.data.uniform_(-0.05, 0.05)
        
    def forward(self, user_id, item_id):
        """
        Forward pass to predict ratings.
        
        Args:
            user_id: Tensor of user indices
            item_id: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Lookup embeddings
        user_vectors = self.user_embedding(user_id)
        item_vectors = self.item_embedding(item_id)
        
        # Compute dot product
        output = torch.sum(user_vectors * item_vectors, dim=1)
        
        return output


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model.
    
    This model combines matrix factorization with a multi-layer perceptron
    to model complex user-item interactions.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[256, 128, 64]):
        """
        Initialize NCF model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Dimension of embedding vectors
            hidden_dims: List of hidden layer dimensions for MLP
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # MF embeddings
        self.mf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings (separate from MF embeddings)
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = 2 * embedding_dim  # Concatenated user and item embeddings
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.output_layer = nn.Linear(hidden_dims[-1] + embedding_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_id, item_id):
        """
        Forward pass to predict ratings.
        
        Args:
            user_id: Tensor of user indices
            item_id: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Matrix Factorization path
        mf_user_vector = self.mf_user_embedding(user_id)
        mf_item_vector = self.mf_item_embedding(item_id)
        mf_vector = mf_user_vector * mf_item_vector  # Element-wise product
        
        # MLP path
        mlp_user_vector = self.mlp_user_embedding(user_id)
        mlp_item_vector = self.mlp_item_embedding(item_id)
        mlp_vector = torch.cat([mlp_user_vector, mlp_item_vector], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        # Concatenate MF and MLP features
        vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        
        # Final prediction
        output = self.output_layer(vector)
        
        return output


class TransformerRecommender(nn.Module):
    """
    Transformer-based recommendation model.
    
    This advanced model uses self-attention mechanisms to capture
    complex patterns in user-item interactions.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, nhead=4, 
                 num_encoder_layers=2, dim_feedforward=256):
        """
        Initialize Transformer recommender model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Dimension of embedding vectors
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network in transformer
        """
        super(TransformerRecommender, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Position encoding (optional for this use case)
        self.pos_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU()
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_id, item_id):
        """
        Forward pass to predict ratings.
        
        Args:
            user_id: Tensor of user indices
            item_id: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)
        
        # Combine embeddings
        combined = torch.cat([user_emb, item_emb], dim=1)
        input_emb = self.pos_encoder(combined)
        
        # Add batch dimension for transformer (batch_size, seq_len=1, embedding_dim)
        input_emb = input_emb.unsqueeze(1)
        
        # Pass through transformer
        transformer_out = self.transformer_encoder(input_emb)
        
        # Remove sequence dimension
        transformer_out = transformer_out.squeeze(1)
        
        # Final prediction
        output = self.output_layer(transformer_out)
        
        return output
    
