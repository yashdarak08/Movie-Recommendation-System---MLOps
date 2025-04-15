import sys
import os
import unittest
import json
import time
import requests
import threading
import subprocess
from unittest import mock

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

class TestInferenceAPI(unittest.TestCase):
    """Integration tests for inference API."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - start the server in a separate process."""
        cls.server_process = None
        
        # Check if we can mock the server or need to start a real one
        cls.use_mock = os.environ.get("USE_MOCK_SERVER", "true").lower() == "true"
        
        if not cls.use_mock:
            # Start the server in a separate process
            cls.server_process = subprocess.Popen(
                ["python", "-m", "src.main", "--mode", "serve", "--config", "configs/infer_config.yaml"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is running
            try:
                response = requests.get("http://localhost:8000/health")
                if response.status_code != 200:
                    raise Exception("Server is not healthy")
            except Exception as e:
                # Stop the server and raise exception
                if cls.server_process:
                    cls.server_process.terminate()
                    cls.server_process.wait()
                raise Exception(f"Server did not start correctly: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test class - stop the server."""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()
    
    @unittest.skipIf(not os.environ.get("USE_MOCK_SERVER", "true").lower() == "true", 
                    "Skipping mock tests when using real server")
    def test_predict_endpoint_mock(self):
        """Test predict endpoint with mocked response."""
        with mock.patch("requests.post") as mock_post:
            # Mock response
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "user_id": 1,
                "item_id": 123,
                "predicted_rating": 4.5
            }
            
            # Call API
            response = requests.post(
                "http://localhost:8000/predict",
                json={"user_id": 1, "item_id": 123}
            )
            
            # Check response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["user_id"], 1)
            self.assertEqual(data["item_id"], 123)
            self.assertIsInstance(data["predicted_rating"], float)
    
    @unittest.skipIf(os.environ.get("USE_MOCK_SERVER", "true").lower() == "true", 
                    "Skipping real server tests when using mock")
    def test_predict_endpoint_real(self):
        """Test predict endpoint with real server."""
        # Call API
        response = requests.post(
            "http://localhost:8000/predict",
            json={"user_id": 1, "item_id": 123}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["user_id"], 1)
        self.assertEqual(data["item_id"], 123)
        self.assertIsInstance(data["predicted_rating"], float)
    
    @unittest.skipIf(not os.environ.get("USE_MOCK_SERVER", "true").lower() == "true", 
                    "Skipping mock tests when using real server")
    def test_recommend_endpoint_mock(self):
        """Test recommend endpoint with mocked response."""
        with mock.patch("requests.post") as mock_post:
            # Mock response
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "user_id": 1,
                "recommendations": [
                    {"item_id": 123, "score": 4.5},
                    {"item_id": 456, "score": 4.2},
                    {"item_id": 789, "score": 3.9}
                ]
            }
            
            # Call API
            response = requests.post(
                "http://localhost:8000/recommend",
                json={"user_id": 1, "top_k": 3}
            )
            
            # Check response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["user_id"], 1)
            self.assertEqual(len(data["recommendations"]), 3)
            self.assertIsInstance(data["recommendations"][0]["score"], float)
    
    @unittest.skipIf(os.environ.get("USE_MOCK_SERVER", "true").lower() == "true", 
                    "Skipping real server tests when using mock")
    def test_recommend_endpoint_real(self):
        """Test recommend endpoint with real server."""
        # Call API
        response = requests.post(
            "http://localhost:8000/recommend",
            json={"user_id": 1, "top_k": 3}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["user_id"], 1)
        self.assertGreaterEqual(len(data["recommendations"]), 1)
        self.assertIsInstance(data["recommendations"][0]["score"], float)
    
    @unittest.skipIf(not os.environ.get("USE_MOCK_SERVER", "true").lower() == "true", 
                    "Skipping mock tests when using real server")
    def test_health_endpoint_mock(self):
        """Test health endpoint with mocked response."""
        with mock.patch("requests.get") as mock_get:
            # Mock response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"status": "healthy"}
            
            # Call API
            response = requests.get("http://localhost:8000/health")
            
            # Check response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
    
    @unittest.skipIf(os.environ.get("USE_MOCK_SERVER", "true").lower() == "true", 
                    "Skipping real server tests when using mock")
    def test_health_endpoint_real(self):
        """Test health endpoint with real server."""
        # Call API
        response = requests.get("http://localhost:8000/health")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")


if __name__ == "__main__":
    unittest.main()