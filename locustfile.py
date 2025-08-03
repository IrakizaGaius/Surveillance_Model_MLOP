from locust import HttpUser, task, between, events
import time
import os

class ModelApiUser(HttpUser):
    wait_time = between(3, 8)  # More realistic: 3-8 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.test_file_path = "data/test/casual_000.wav"
        if not os.path.exists(self.test_file_path):
            print(f"Warning: Test file {self.test_file_path} not found")
            self.test_file_path = None

    @task(3)  # 60% of requests
    def predict_audio(self):
        """Test audio prediction endpoint"""
        if not self.test_file_path:
            return
            
        try:
            with open(self.test_file_path, "rb") as f:
                files = {"file": ("test.wav", f, "audio/wav")}
                response = self.client.post("/predict", files=files, timeout=30)
                
                if response.status_code == 200:
                    self.environment.events.request.fire(
                        request_type="POST",
                        name="/predict",
                        response_time=response.elapsed.total_seconds() * 1000,
                        response_length=len(response.content),
                        exception=None,
                    )
                else:
                    print(f"Predict failed with status {response.status_code}")
                    
        except Exception as e:
            print(f"Error in predict_audio: {e}")

    @task(2)  # 40% of requests
    def check_status(self):
        """Test status and health endpoints"""
        endpoints = ["/status", "/health"]
        
        for endpoint in endpoints:
            try:
                response = self.client.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    self.environment.events.request.fire(
                        request_type="GET",
                        name=endpoint,
                        response_time=response.elapsed.total_seconds() * 1000,
                        response_length=len(response.content),
                        exception=None,
                    )
                else:
                    print(f"{endpoint} failed with status {response.status_code}")
                    
            except Exception as e:
                print(f"Error in {endpoint}: {e}")

class HeavyLoadUser(HttpUser):
    """User class for stress testing"""
    wait_time = between(1, 3)  # Faster requests for stress testing
    
    def on_start(self):
        self.test_file_path = "data/test/casual_000.wav"
        if not os.path.exists(self.test_file_path):
            self.test_file_path = None

    @task
    def stress_predict(self):
        """Continuous prediction requests for stress testing"""
        if not self.test_file_path:
            return
            
        try:
            with open(self.test_file_path, "rb") as f:
                files = {"file": ("test.wav", f, "audio/wav")}
                response = self.client.post("/predict", files=files, timeout=30)
                
                if response.status_code != 200:
                    print(f"Stress test failed: {response.status_code}")
                    
        except Exception as e:
            print(f"Stress test error: {e}")

# Event listeners for better monitoring
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    print("🚀 Load testing initialized")
    print("📊 Available user classes:")
    print("  - ModelApiUser: Realistic user behavior (3-8s between requests)")
    print("  - HeavyLoadUser: Stress testing (1-3s between requests)")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("🔥 Load test starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("✅ Load test completed")
