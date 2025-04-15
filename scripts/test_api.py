#!/usr/bin/env python
"""
Simple script to test the recommendation API.
"""

import argparse
import json
import time
import requests
from typing import Dict, List, Any


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
        return False


def test_predict(base_url: str, user_id: int, item_id: int) -> bool:
    """Test predict endpoint."""
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"user_id": user_id, "item_id": item_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful: {data['predicted_rating']:.2f}")
            print(f"   User ID: {data['user_id']}")
            print(f"   Item ID: {data['item_id']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Prediction failed with error: {e}")
        return False


def test_batch_predict(base_url: str, user_ids: List[int], item_ids: List[int]) -> bool:
    """Test batch predict endpoint."""
    try:
        response = requests.post(
            f"{base_url}/batch_predict",
            json={"user_ids": user_ids, "item_ids": item_ids}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful")
            print(f"   Latency: {data['latency_ms']:.2f} ms")
            print(f"   Results:")
            for i, pred in enumerate(data['predictions']):
                print(f"     - User {pred['user_id']}, Item {pred['item_id']}: {pred['predicted_rating']:.2f}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Batch prediction failed with error: {e}")
        return False


def test_recommend(base_url: str, user_id: int, top_k: int = 5) -> bool:
    """Test recommend endpoint."""
    try:
        response = requests.post(
            f"{base_url}/recommend",
            json={"user_id": user_id, "top_k": top_k}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recommendations successful")
            print(f"   User ID: {data['user_id']}")
            print(f"   Latency: {data['latency_ms']:.2f} ms")
            print(f"   Top {len(data['recommendations'])} recommendations:")
            
            for i, rec in enumerate(data['recommendations']):
                title = rec.get('title', 'Unknown')
                print(f"     {i+1}. {title} (ID: {rec['item_id']}) - Score: {rec['score']:.2f}")
            
            return True
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Recommendations failed with error: {e}")
        return False


def performance_test(base_url: str, user_id: int, num_requests: int = 100) -> bool:
    """Test API performance."""
    print(f"🔄 Running performance test with {num_requests} sequential requests...")
    
    start_time = time.time()
    success_count = 0
    latencies = []
    
    for i in range(num_requests):
        try:
            req_start = time.time()
            response = requests.post(
                f"{base_url}/predict",
                json={"user_id": user_id, "item_id": 1 + (i % 100)}
            )
            req_latency = (time.time() - req_start) * 1000  # ms
            latencies.append(req_latency)
            
            if response.status_code == 200:
                success_count += 1
            
            if i % 10 == 0 and i > 0:
                print(f"   Completed {i}/{num_requests} requests...")
                
        except Exception as e:
            print(f"   Request {i+1} failed: {e}")
    
    total_time = time.time() - start_time
    success_rate = (success_count / num_requests) * 100
    
    print(f"✅ Performance test completed")
    print(f"   Success rate: {success_rate:.2f}%")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Throughput: {num_requests / total_time:.2f} requests/second")
    print(f"   Avg latency: {sum(latencies) / len(latencies):.2f} ms")
    print(f"   P50 latency: {sorted(latencies)[len(latencies)//2]:.2f} ms")
    print(f"   P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms")
    print(f"   P99 latency: {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms")
    
    return success_rate > 95  # Consider test passed if success rate > 95%


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test recommendation API")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--user-id", type=int, default=1, help="User ID for recommendations")
    parser.add_argument("--item-id", type=int, default=1, help="Item ID for prediction")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests for performance test")
    parser.add_argument("--test", type=str, choices=["all", "health", "predict", "batch", "recommend", "performance"],
                        default="all", help="Test to run")
    
    args = parser.parse_args()
    
    print("🚀 Starting API tests")
    print(f"   URL: {args.url}")
    print(f"   User ID: {args.user_id}")
    print(f"   Item ID: {args.item_id}")
    print()
    
    if args.test == "all" or args.test == "health":
        test_health(args.url)
        print()
    
    if args.test == "all" or args.test == "predict":
        test_predict(args.url, args.user_id, args.item_id)
        print()
    
    if args.test == "all" or args.test == "batch":
        user_ids = [args.user_id] * 5
        item_ids = [args.item_id, args.item_id + 1, args.item_id + 2, args.item_id + 3, args.item_id + 4]
        test_batch_predict(args.url, user_ids, item_ids)
        print()
    
    if args.test == "all" or args.test == "recommend":
        test_recommend(args.url, args.user_id)
        print()
    
    if args.test == "all" or args.test == "performance":
        performance_test(args.url, args.user_id, args.num_requests)
        print()
    
    print("🏁 All tests completed")


if __name__ == "__main__":
    main()