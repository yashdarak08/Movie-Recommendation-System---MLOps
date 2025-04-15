import sys
import os
import unittest
import torch
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models import MatrixFactorizationModel, NeuralCollaborativeFiltering, TransformerRecommender, get_model


class TestRecommenderModels(unittest.TestCase):
    """Test cases for recommendation models."""
    
    def setUp(self):
        """Set up test cases."""
        self.num_users = 100
        self.num_items = 50
        self.embedding_dim = 32
        self.batch_size = 16
        
        # Create random user and item indices
        self.user_indices = torch.randint(0, self.num_users, (self.batch_size,))
        self.item_indices = torch.randint(0, self.num_items, (self.batch_size,))
    
    def test_matrix_factorization_model(self):
        """Test Matrix Factorization model."""
        model = MatrixFactorizationModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim
        )
        
        # Test forward pass
        predictions = model(self.user_indices, self.item_indices)
        
        # Check output shape and type
        self.assertEqual(predictions.shape, (self.batch_size,))
        self.assertTrue(torch.is_floating_point(predictions))
    
    def test_ncf_model(self):
        """Test Neural Collaborative Filtering model."""
        model = NeuralCollaborativeFiltering(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            hidden_dims=[64, 32, 16]
        )
        
        # Test forward pass
        predictions = model(self.user_indices, self.item_indices)
        
        # Check output shape
        self.assertEqual(predictions.shape, (self.batch_size, 1))
        self.assertTrue(torch.is_floating_point(predictions))
    
    def test_transformer_model(self):
        """Test Transformer model."""
        model = TransformerRecommender(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            nhead=2
        )
        
        # Test forward pass
        predictions = model(self.user_indices, self.item_indices)
        
        # Check output shape
        self.assertEqual(predictions.shape, (self.batch_size, 1))
        self.assertTrue(torch.is_floating_point(predictions))
    
    def test_get_model_function(self):
        """Test get_model function."""
        config = {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim
        }
        
        # Test for matrix factorization
        mf_model = get_model("mf", config)
        self.assertIsInstance(mf_model, MatrixFactorizationModel)
        
        # Test for neural collaborative filtering
        ncf_model = get_model("ncf", config)
        self.assertIsInstance(ncf_model, NeuralCollaborativeFiltering)
        
        # Test for transformer
        transformer_model = get_model("transformer", config)
        self.assertIsInstance(transformer_model, TransformerRecommender)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            get_model("invalid_model", config)


if __name__ == "__main__":
    unittest.main()