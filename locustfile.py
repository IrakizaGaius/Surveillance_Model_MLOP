from locust import HttpUser, task, between

class ModelApiUser(HttpUser):
    wait_time = between(1, 2)  # seconds between requests

    @task
    def predict(self):
        # Adjust the endpoint and payload to match your API
        files = {
            "file": ("test.wav", open("data/test/casual_000.wav", "rb"), "audio/wav")
        }
        self.client.post("/predict", files=files)
